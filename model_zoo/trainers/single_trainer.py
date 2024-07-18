import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision.utils
import typing as th



class BaseTrainer:
    """Base class for SingleTrainer and AlternatingEpochTrainer"""
    _STEPS_PER_LOSS_WRITE = 10

    def __init__(
        self,

        module, *,
        ckpt_prefix,

        train_loader,
        valid_loader,
        test_loader,

        writer,

        max_epochs,

        early_stopping_metric=None,
        max_bad_valid_epochs=None,
        max_grad_norm=None,
        
        sample_freq: th.Optional[int] = None,
        progress_bar: th.Optional[bool] = True,
    ):
        self.module = module
        self.ckpt_prefix = ckpt_prefix

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.writer = writer

        self.max_epochs = max_epochs

        self.early_stopping_metric = early_stopping_metric
        self.max_grad_norm = max_grad_norm
        self.bad_valid_epochs = 0
        self.best_valid_loss = math.inf

        if max_bad_valid_epochs is None:
            self.max_bad_valid_epochs = math.inf
        else:
            self.max_bad_valid_epochs = max_bad_valid_epochs

        self.iteration = 0
        self.epoch = 0


        
        self.sample_freq = sample_freq
        self.progress_bar = progress_bar

    def train(self):

        self.update_transform_parameters()

        while self.epoch < self.max_epochs and self.bad_valid_epochs < self.max_bad_valid_epochs:
            self.module.train()

            self.train_for_epoch()
            
            with torch.no_grad():
                valid_loss = self.module.loss(self.valid_loader).mean()
            
            # TODO: something weird is happening here
            if self.early_stopping_metric is not None and self.early_stopping_metric != 'None':
                if valid_loss < self.best_valid_loss:
                    self.bad_valid_epochs = 0
                    self.best_valid_loss = valid_loss
                    self.write_checkpoint("best_valid")

                    print(f"Best validation loss of {valid_loss} achieved on epoch {self.epoch}")

                else:
                    self.bad_valid_epochs += 1

                    if self.bad_valid_epochs == self.max_bad_valid_epochs:
                        print(f"No validation improvement for {self.max_bad_valid_epochs}"
                                + " epochs. Training halted.")
                        self.write_checkpoint("latest")

                        self.load_checkpoint("best_valid")

                        return

            self.write_checkpoint("latest")
            
            if self.sample_freq is not None and self.epoch % self.sample_freq == 0:
                self.sample_and_record()
                
            # write the epoch itself
            self.write_scalar("epoch", self.epoch - 1, None)

        if self.bad_valid_epochs < self.max_bad_valid_epochs:
            print(f"Maximum epochs reached. Training of {self.module.model_type} complete.")

    def train_for_epoch(self):
        if self.progress_bar:
            pbar = self._tqdm_progress_bar(
                iterable=enumerate(self.train_loader),
                desc="Training",
                length=len(self.train_loader),
                leave=True
            )
        else:
            pbar = enumerate(self.train_loader)
            
        for j, (batch, _, idx) in pbar:
            loss_dict = self.train_single_batch(batch)

            if j == 0:
                full_loss_dict = loss_dict
            else:
                for k in loss_dict.keys():
                    full_loss_dict[k] += loss_dict[k]

        self.epoch += 1
        self.update_transform_parameters()

        for k, v in full_loss_dict.items():
            print(f"{self.module.model_type} {k}: {v/j:.4f} after {self.epoch} epochs")

    def _test(self):
        if len(self.module.data_shape) > 1: # If image data
            self.sample_and_record()

        test_results = self.evaluator.test()
        self.record_dict(self.ckpt_prefix + "_test", test_results, self.epoch, save=True)

    def _validate(self):
        valid_results = self.evaluator.validate()
        self.record_dict("validate", valid_results, self.epoch)
        return valid_results.get(self.early_stopping_metric)

    def update_transform_parameters(self):
        raise NotImplementedError("Define in child classes")

    def _tqdm_progress_bar(self, iterable, desc, length, leave):
        return tqdm(
            iterable,
            desc=desc,
            total=length,
            bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
            leave=leave
        )

    def write_scalar(self, tag, value, step: th.Optional[int] = None):
        self.writer.write_scalar(f"{self.module.model_type}/{tag}", value, step)

    def sample_and_record(self):
        NUM_SAMPLES = 9
        GRID_ROWS = 3

        with torch.no_grad():
            try:
                imgs = self.module.sample(NUM_SAMPLES)
            except AttributeError:
                print("No sample method available")
                return

            imgs.clamp_(self.module.data_min, self.module.data_max)
            grid = torchvision.utils.make_grid(imgs, nrow=GRID_ROWS, pad_value=1, normalize=True, scale_each=True)

            self.writer.write_image("samples", grid, global_step=self.epoch)

    def record_dict(self, tag_prefix, value_dict, step, save=False):
        for k, v in value_dict.items():
            print(f"{self.module.model_type} {k}: {v:.4f}")
            self.write_scalar(f"{tag_prefix}/{k}", v, step)

        if save:
            self.writer.write_json(
                f"{tag_prefix}_{self.module.model_type}_metrics",
                {k: v.item() for k, v in value_dict.items()}
            )

    def write_checkpoint(self, tag):
        raise NotImplementedError("Define in child classes")

    def load_checkpoint(self, tag):
        raise NotImplementedError("Define in child classes")

    def _get_checkpoint_name(self, tag):
        return f"{self.ckpt_prefix}_{self.module.model_type}_{tag}"


class SingleTrainer(BaseTrainer):
    """Class for training single module"""

    def train_single_batch(self, batch):
        
        loss_dict = self.module.train_batch(batch, max_grad_norm=self.max_grad_norm, trainer=self)

        if self.iteration % self._STEPS_PER_LOSS_WRITE == 0:
            for k, v in loss_dict.items():
                self.write_scalar("train/"+k, v, self.iteration+1)

        self.iteration += 1
        return loss_dict

    def update_transform_parameters(self):
        train_dset = self.train_loader.dataset

        self.module.data_min = torch.ones_like(self.module.data_min) * train_dset.get_data_min()
        self.module.data_max = torch.ones_like(self.module.data_max) * train_dset.get_data_max()
        self.module.data_shape = train_dset.get_data_shape()


        if self.module.whitening_transform:
            self.module.set_whitening_params(
                torch.mean(train_dset.tensorize_and_concatenate_all(), dim=0, keepdim=True),
                torch.std(train_dset.tensorize_and_concatenate_all(), dim=0, keepdim=True)
            )

    def write_checkpoint(self, tag):
        if self.module.num_optimizers == 1:
            opt_state_dict = self.module.optimizer.state_dict()
            lr_state_dict = self.module.lr_scheduler.state_dict()
        else:
            opt_state_dict = [opt.state_dict() for opt in self.module.optimizer]
            lr_state_dict = [lr.state_dict() for lr in self.module.lr_scheduler]

        checkpoint = {
            "iteration": self.iteration,
            "epoch": self.epoch,

            "module_state_dict": self.module.state_dict(),
            "opt_state_dict": opt_state_dict,
            "lr_state_dict": lr_state_dict,

            "bad_valid_epochs": self.bad_valid_epochs,
            "best_valid_loss": self.best_valid_loss
        }

        self.writer.write_checkpoint(self._get_checkpoint_name(tag), checkpoint)

    def load_checkpoint(self, tag):
        try:
            checkpoint = self.writer.load_checkpoint(self._get_checkpoint_name(tag), self.module.device)
        except FileNotFoundError as e:
            print("[WARNING] No checkpoint available!")
            return
        
        self.iteration = checkpoint["iteration"]
        self.epoch = checkpoint["epoch"]

        self.module.load_state_dict(checkpoint["module_state_dict"])

        if self.module.num_optimizers == 1:
            self.module.optimizer.load_state_dict(checkpoint["opt_state_dict"])
            try:
                self.module.lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
            except KeyError:
                print("WARNING: Not setting lr scheduler state dict since it is not available in checkpoint.")
        else:
            for (optimizer, state_dict) in zip(self.module.optimizer, checkpoint["opt_state_dict"]):
                optimizer.load_state_dict(state_dict)
            try:
                for (lr_scheduler, state_dict) in zip(self.module.lr_scheduler, checkpoint["lr_state_dict"]):
                    lr_scheduler.load_state_dict(state_dict)
            except KeyError:
                print("WARNING: Not setting lr scheduler state dict since it is not available in checkpoint.")

        self.bad_valid_epochs = checkpoint["bad_valid_epochs"]
        self.best_valid_loss = checkpoint["best_valid_loss"]

        print(f"Loaded {self.module.model_type} checkpoint `{tag}' after epoch {self.epoch}")

    def get_all_loaders(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def update_all_loaders(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # TODO: if there are problems with train_loader in evaluator, consider refactoring
        #       evaluator to include train_loader as an attribute
        self.evaluator.valid_loader = valid_loader
        self.evaluator.test_loader = test_loader
