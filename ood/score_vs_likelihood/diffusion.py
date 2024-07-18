
class DiffusionScoreNormVsLikelihoodOODDetection(OODBaseMethod):
    """
    score-based ood detection method for diffusion model
    """
    def __init__(
        self,
        
        # The basic parameters passed to any OODBaseMethod
        likelihood_model: ScoreBasedDiffusion,    
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        checkpoint_dir: th.Optional[str] = None,
        checkpointing_buffer: th.Optional[int] = None,
        
        # The intrinsic dimension calculator args
        log_prob_kwargs: th.Optional[th.Dict[str, th.Any]] = None,
        # for logging args
        verbose: int = 0,
        
        # Hyper-parameters relating to the scale parameter that is being computed
        evaluate_eps: float = 0.00001,
    ):
        self.likelihood_model = likelihood_model
        super().__init__(
            x_loader=x_loader, 
            likelihood_model=likelihood_model, 
            in_distr_loader=in_distr_loader, 
            checkpoint_dir=checkpoint_dir,
        )
        self.checkpointing_buffer = checkpointing_buffer
        self.verbose = verbose
        self.evaluate_eps = evaluate_eps
        
        self.log_prob_kwargs = log_prob_kwargs or {}
    
    def _get_loader_dimensionality(
        self,
        r: float,
        loader,
    ):
        all_lids = []
        for x in loader:
            all_lids.append(
                self.likelihood_model.lid(
                    x,
                    r=r,
                    **self.intrinsic_dimension_calculator_args,
                )
            )
        all_lids = torch.cat(all_lids, dim=0)
        return all_lids.cpu().numpy()
        
    def run(self):
        
        
        progress_dict = {}
        if self.checkpoint_dir is not None and os.path.exists(os.path.join(self.checkpoint_dir, 'progress.json')):
            with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'r') as file:
                progress_dict = json.load(file)
        
        
        # All score-norms and all likelihoods update
        all_score_norms= None
        all_likelihoods = None
        if self.checkpoint_dir is not None:
            if os.path.exists(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy')):
                all_likelihoods = np.load(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy'), allow_pickle=True)
            if os.path.exists(os.path.join(self.checkpoint_dir, 'all_score_norms.npy')):
                all_score_norms = np.load(os.path.join(self.checkpoint_dir, 'all_score_norms.npy'), allow_pickle=True)
        
        idx = 0
        
        
        for inner_loader in buffer_loader(self.x_loader, self.checkpointing_buffer):
            idx += 1
            if 'buffer_progress' in progress_dict:
                if idx <= progress_dict['buffer_progress']:
                    continue
            if self.verbose > 0:
                print(f"Working with buffer [{idx}]")
            
            
            # Compute and add score norms
            if self.verbose > 0:
                print("Computing score norms ... ")
            
            all_buffer_score_norms = None
            inner_loader_rng = tqdm(inner_loader, total=len(inner_loader)) if self.verbose > 0 else inner_loader
            for x in inner_loader_rng:
                with torch.no_grad():
                    eps = torch.tensor(self.evaluate_eps).float().to(x.device)
                    score_norms = torch.linalg.norm(self.likelihood_model._get_unnormalized_score(x, eps).reshape(x.shape[0], -1), dim=1).flatten().cpu().numpy()
                    all_buffer_score_norms = np.concatenate([all_buffer_score_norms, score_norms]) if all_buffer_score_norms is not None else score_norms
            
            # compute and add likelihoods
            if self.verbose > 0:
                print("Computing likelihoods ... ")
                
            all_buffer_likelihoods = None
            inner_loader_rng = tqdm(inner_loader, total=len(inner_loader)) if self.verbose > 0 else inner_loader
            for x in inner_loader_rng:
                with torch.no_grad():
                    likelihoods = self.likelihood_model.log_prob(x, **self.log_prob_kwargs).cpu().numpy().flatten()
                    all_buffer_likelihoods = np.concatenate([all_buffer_likelihoods, likelihoods]) if all_buffer_likelihoods is not None else likelihoods
                    
            all_score_norms = np.concatenate([all_score_norms, all_buffer_score_norms]) if all_score_norms is not None else all_buffer_score_norms
            all_likelihoods = np.concatenate([all_likelihoods, all_buffer_likelihoods]) if all_likelihoods is not None else all_buffer_likelihoods
            
            
            progress_dict['buffer_progress'] = idx
            
            if self.checkpoint_dir is not None:
                np.save(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy'), all_likelihoods)
                np.save(os.path.join(self.checkpoint_dir, 'all_score_norms.npy'), all_score_norms)
                
                with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'w') as file:
                    json.dump(progress_dict, file)
            
            
            

        visualize_scatterplots(
            scores = np.stack([all_likelihoods, all_score_norms]).T,
            column_names=["log-likelihood", "score-norm"],
        )