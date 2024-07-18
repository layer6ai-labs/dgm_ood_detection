from .base_method import OODBaseMethod
import torch
import typing as th
import dypy as dy
import wandb
from tqdm import tqdm

class OODLocalOptimization(OODBaseMethod):
    """
    This class of methods does a local optimization on OOD samples
    in terms of the input.
    
    Showing some properties such as Negative Log Likelihood (NLL) and
    the gradient of NLL with respect to the input will give us some
    insights into the behavior of the likelihood model.
    
    The intention is to find a point that is close to x and has a high likelihood which is
    peaked.
    """

    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        optimization_steps: int,
        optimizer: th.Union[torch.optim.Optimizer, str],
        representation_rank: th.Optional[int] = None,
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        logger: th.Optional[th.Any] = None,
        optimizer_args: th.Optional[th.Dict[str, th.Any]] = None,
        optimization_objective: th.Literal["nll", "grad_norm"] = "nll",
        intermediate_image_log_frequency: th.Optional[int] = None,
        pick_best: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(x=x, x_batch=x_batch, likelihood_model=likelihood_model, logger=logger, **kwargs)
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.optimization_objective = optimization_objective
        self.intermediate_image_log_frequency = intermediate_image_log_frequency
        self.pick_best = pick_best
        self.representation_rank = representation_rank
        if self.x_batch is not None:
            raise NotImplementedError("x_batch is not implemented yet.")

    def run(self):
        """
        This function performs a local optimization on the input x to the likelihood model.

        The intention is to find a point that is close to x and has a high likelihood which is
        peaked. However, plugging in a tensorboard_writer will also tell you how the likelihood
        function behaves close to x. This is useful for getting insights into the first and second
        order behavior of the likelihood function.

        Args:
            x (torch.Tensor): The input
            likelihood_model (torch.nn.Module): The likelihood parametric model
            optimization_steps (int): Number of steps you want the optimization to run
            optimizer (th.Union[torch.optim.Optimizer, str]):
                The class name of the optimizer to use as a string or as the class itself
            optimizer_args (th.Optional[th.Dict[str, th.Any]], optional):
                The arguments for the optimizer to be instantiated.
                Defaults to None which means the empty dictionary.
            optimization_objective:
                The objective is either the negative log likelihood (-model.log_prob)
                or the norm of the gradient of the nll, i.e., || grad -model.log_prob ||
                Defaults to "nll".
            tensorboard_writer (th.Optional[tensorboard.SummaryWriter], optional):
                You can pass a tensorboard summary writer to this one as well that summarizes the following:
                1. The distance from origin of the image
                2. The beginning image itself
                3. The image it ends up at.
                4. The gradient norms of nll
                5. The nll itself
                Defaults to None, which means that none of these will be logged.
            pick_best (bool, optional):
                If set to True, the model picks the intermediate input that produces
                the smallest objective.
                Defaults to True.
        Returns:
            self.repr_point: Manipulates x so that self.repr_point is the point that is peaked in terms of the model likelihood.
        """
        # TODO: a bit of refactoring is needed here
        likelihood_model = self.likelihood_model
        device = likelihood_model.device

        # unsqueeze x and prepare it for giving it as input
        if self.x is None:
            raise ValueError("x is None. Please provide a value for x.")
        x = self.x.unsqueeze(0).to(device)
        if self.x_batch is not None:
            raise NotImplementedError("x_batch is not implemented yet.")
        
        optimization_steps = self.optimization_steps
        optimizer = self.optimizer
        optimizer_args = self.optimizer_args
        optimization_objective = self.optimization_objective
        intermediate_image_log_frequency = self.intermediate_image_log_frequency
        pick_best = self.pick_best

        # add the original image to the tensorboard writer
        if self.logger is not None:
            self.logger.log(
                {"optimization/original_image": [wandb.Image(x, caption="original image we start off with")]}
            )

        # turn off all the gradients for likelihood_model
        for param in self.likelihood_model.parameters():
            param.requires_grad_(False)
        
        if self.representation_rank is not None:
            # run the likelihood model on x once to get the representation
            # likelihood_model.clear_representation()
            
            repr_module = likelihood_model.get_representation_module(self.representation_rank)
            
            def hook_fn1(module, args, output):
                self.repr_point = output[0].clone().detach()
            handle = repr_module.register_forward_hook(hook_fn1)
            # call the log_prob function just to store the representation in self.repr_point
            likelihood_model.log_prob(x)
            # remove the hook
            handle.remove()
            
            self.repr_point.requires_grad = True
            
            def hook_fn(module, args, output):
                # return a tuple where the first element is self.rep_point
                # and the rest of the elements are output[1:] that are detached
                # from the graph
                return (self.repr_point,) + tuple([x.clone().detach() for x in output[1:]])
            
            repr_module = likelihood_model.get_representation_module(self.representation_rank)
            handle = repr_module.register_forward_hook(hook_fn)
            
            origin = self.repr_point.clone()
        else:    
            # TODO: handle the issue here with LBFGS
            self.repr_point = x.clone().detach().contiguous()
            self.repr_point.requires_grad = True
            
            likelihood_model = torch.nn.Sequential(torch.nn.Identity(), likelihood_model)
            # add a method log_prob to the likelihood model which is equal to the log_prob of the second part
            likelihood_model.log_prob = lambda x: likelihood_model[1].log_prob(likelihood_model[0](x))
            
            # get the first part of the likelihood model and register a hook to get the representation
            def hook_fn(module, args, output):
                return self.repr_point
            
            handle = likelihood_model[0].register_forward_hook(hook_fn)

            origin = x.clone()
            
        def forward_func(x):
            # get the device of the torch.nn.Module likelihood_model
            return -likelihood_model.log_prob(x).squeeze(0)

        optimizer_args = optimizer_args or {}

        # change x to self.repr_point by optimizing the input according to the objective
        optimizer = (
            optimizer([self.repr_point], **optimizer_args)
            if isinstance(optimizer, torch.optim.Optimizer)
            else dy.eval(optimizer)([self.repr_point], **optimizer_args)
        )

        # the objective is either the negative log likelihood or the gradient norm
        if optimization_objective == "nll":
            obj_func = lambda x: forward_func(x)
        elif optimization_objective == "grad_norm":
            obj_func = lambda x: torch.norm(torch.func.vjp(forward_func, x)[0])
        else:
            raise Exception(
                "optimization_objective must be specified!\n" "Please choose between 'nll' and 'grad_norm'"
            )

        # do the optimization and keep the place where the gradient
        # is the smallest for the point to be a peak
        best_repr_point = None
        best_obj = None

        if self.progress_bar:
            iterable = tqdm(range(optimization_steps))
        else:
            iterable = range(optimization_steps)
            
        for _ in iterable:
            if isinstance(optimizer, torch.optim.LBFGS):
                # Do a second-order optimization with closure
                def closure():
                    optimizer.zero_grad()
                    loss = obj_func(x)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
            else:
                # Do a simple first order optimization
                optimizer.zero_grad()
                loss = obj_func(x)
                loss.backward()
                optimizer.step()
            
            nll = forward_func(x)
            
            nll.backward()
            nll_grad_norm = torch.norm(self.repr_point.grad, p=1.0) / self.repr_point.numel()
            
            # clear the gradient
            self.repr_point.grad.zero_()
            
            if pick_best:
                with torch.no_grad():
                    t = obj_func(x)
                    if best_obj is None or t < best_obj:
                        best_obj = t
                        best_repr_point = self.repr_point.clone()

            if self.logger is not None:
                with torch.no_grad():
                    NLL_log = nll.detach().cpu().item()
                    objective = obj_func(x).detach().cpu().item()
                    self.logger.log(
                        {
                            "nll_grad_norm": nll_grad_norm,
                            "nll": NLL_log,
                            "||x - origin||": torch.norm(self.repr_point - origin),
                            "objective": objective,
                        }
                    )
                if intermediate_image_log_frequency is not None and (_ + 1) % intermediate_image_log_frequency == 0:
                    # if self.repr_point has a representation [c, h, w], then split the channels
                    # into three almost equal parts and average each of the parts to get a [3, h, w] image
                    # that we can log to tensorboard
                    repr_output = True
                    
                    img = self.repr_point.clone().detach().cpu().squeeze(0)
                    
                    
                    if len(img.shape) == 3:
                        repr_output = False
                        if img.shape[1] != img.shape[2] or img.shape[1] == 1:
                            img = img.reshape(-1)
                            repr_output = True
                        
                    if repr_output:
                        img = img.reshape(-1)
                        # by zero padding make img.shape[1] divisible by 32 * 32
                        img = torch.nn.functional.pad(img, (0, 32 * 32 - img.shape[0] % (32 * 32)))
                        img = img.reshape(-1, 32, 32)
                    
                    
                    if img.shape[0] > 3:
                        first_third = img[: img.shape[0] // 3, :, :].mean(dim=0, keepdim=True)
                        second_third = img[img.shape[0] // 3 : 2 * img.shape[1] // 3, :, :].mean(
                            dim=0, keepdim=True
                        )
                        third_third = img[2 * img.shape[0] // 3 :, :, :].mean(dim=0, keepdim=True)
                        img = torch.cat([first_third, second_third, third_third], dim=0)
                    elif img.shape[1] == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.shape[1] == 2:
                        img = torch.cat([img, img[0, :, :]], dim=0)
                        
                    try:
                        self.logger.log(
                            {
                                "optimization/intermediate_representation": [
                                    wandb.Image(img, caption=f"The latent representation.")
                                ]
                            }
                        )
                    except Exception as e:
                        print(e)
                        print("Could not log intermediate image.")

        if pick_best:
            self.repr_point = best_repr_point

        # turn off all the gradients for likelihood_model
        for param in self.likelihood_model.parameters():
            param.requires_grad_(True)
        
        # remove the hook that helped us get the representation
        handle.remove()
        
        return self.repr_point.detach().cpu()
