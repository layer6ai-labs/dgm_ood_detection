
# Configuration Guide

We have a systematic hierarchical configuration format that we use for all our experiments, we use this convention for better version control, transparency, and fast and extendable development. Every detail of model training and OOD detection is encapsulated within `yaml` file, which are then parsed using the [`jsonargparse`]() library for running the runnable files.

## Single Training Configurations

These configurations relate to training specific models (non-two-step models). Examples of such configurations can be found in [this](configuration/training) directory. You can run models with such defined configurations using the following command:

```bash
python train.py --config <path-to-single-training-configuration>
```

Now let us run through the specifications of what a training yaml would look like using the following example:

```yaml
# (1)   The configurations pertaining to the dataset being used for training
data:
    # A list of all the datasets that are supported is given in:
    # model_zoo/datasets/utils.py
    dataset: <name-of-the-dataset>
    train_batch_size: <batch-size-for-training>
    test_batch_size: <batch-size-for-testing>
    valid_batch_size: <batch-size-for-validation>
    # Note that all the subconfigs will be passed directly to the
    # model_zoo/datasets/loaders.py and the get_loaders function
    # so feel free to define any extra arguments here
    embedding_network: <name-of-the-embedder>
    # Here, the embedding network name is define.
    # For example, you can define an NVIDIA_efficientnet_b0 here.

# (2)   The configurations pertaining to the model being used for training
#       This is typically a torch.nn.Module
model:
    # The model class, all should inherit from the base class defined in
    # GeneralizedAutoEncoder or DensityEstimator
    # These models have certain properties such as a log_prob function
    # some optimizer defined, etc.
    class_path: <path-to-the-model-class>
    # Please refer to code documentation for the specified class to 
    # see the appropriate arguments
    init_args: <dictionary-of-the-arguments>

# (3)   The configurations pertaining to the optimizer being used for training
trainer:
    # The class of the trainer, an example is 
    # model_zoo.trainers.single_trainer.SingleTrainer
    # All of these classes should inherit from the base class defined in
    # *model_zoo.trainers.single_trainer.BaseTrainer*
    trainer_cls: <class-of-the-trainer>
    # configurations relating to the optimizer
    optimizer:
        # the class of the optimizer, an example is torch.optim.AdamW
        class_path: <torch-optimizer-class>
        # you can define the base lr here for example
        init_args:
            lr: <learning-rate>
            # additional init args for an optimizer

        # a scheduler used on top of the given optimizer
        lr_scheduler:
            # the class of the scheduler, an example is torch.optim.lr_scheduler.ExponentialLR
            class_path: <torch-scheduler-class>
            init_args: 
                gamma: <gamma>
                # additional init args for the scheduler

    writer: <dictionary-of-arguments>
    # all the arguments given to the writer class
    # being used. You can check the arguments
    # in the init arguments of the class
    # *model_zoo.writer.Writer*
    # NOTE: the wandb is only supported now
        
    evaluator:
        valid_metrics: [loss]
        test_metrics: [loss]

    sample_freq: <x> # logs the samples after every x epochs (only works for images, set null otherwise)
    max_epochs: <x> # The number of epochs to run
    early_stopping_metric: loss # The metric used for early stopping on the validation
    max_bad_valid_epochs: <x> # The maximum number of bad validations needed for early stopping
    max_grad_norm: <x> # gradient clipping
    progress_bar: <bool> # output a progress bar or not
```

## OOD Detection Configurations

The following is an example of a configuration defined for OOD detection. Examples of such configurations exist in [this]() directory.

```yaml
# (1)   The configurations pertaining to the likelihood model being used for OOD detection
base_model:
    # A directory containing the configuration used for instantiating the likelihood model itself
    # it can be a json or a yaml file
    config_dir: <path-to-json-or-yaml-file>
    # A directory containing the model weights of the likelihood model to work with a good fit one
    # This is produced by running the single step training models
    checkpoint_dir: <path-to-checkpoints>

# (2)   The configurations pertaining to the dataset pairs that are used for OOD detection
#       the first dataset is the in-distribution dataset and the second one is the out-of-distribution
data:
    # specify the datasets and dataloader configurations for the in and out of distribution data.
    in_distribution:
        pick_loader: <train/test/valid> # The loader to pick between the created loaders for OOD detection
        dataloader_args: <dataloader-args-for-in-distribution>
        # A dictionary defined here is directly passed to the get_loaders function in model_zoo/datasets/loaders.py

    out_of_distribution:
        pick_loader: <train/test/valid> # The loader to pick between the created loaders for in-distribution
        dataloader_args: <dataloader-args-for-out-of-distribution>
        # A dictionary defined here is directly passed to the get_loaders function in model_zoo/datasets/loaders.py
    
    # for reproducibility of the shuffling of the datasets
    seed: <x>

ood:
    # configurations related to visualizing the data in consideration before even starting
    # to perform the OOD detection task
    <visualization-arguments-and-corresponding-keys>
    
    # Batch/Loader/Datapoint
    # OOD detection algorithms operate over either single datapoints, batches, or on entire loaders
    # Using the pick_single variable, you can specify whether you want to pick a single datapoint or not
    # if pick_single is set to false, you can specify whether you want to perform OOD detection on the entire
    # dataloader or not.
    <OOD-data-configuration>

    # for reproducibility of the shuffling of the datasets
    seed: <x>
    
    # An ood method class that inherits from the base class defined in
    # ood.base_method.OODMethodBaseClass
    method: <method-class>
    method_args: <dictionary-being-passed-for-initialization>
        

# NOTE: only supported for Weights & Biases at the moment
logger:
    project: <W&B-project>
    entity: <W&B-entity>
    name: <name-of-the-run>
```

**Different datasets for OOD detection**: The datasets are defined using the hierarchical config in `<dataloader-args-for-in-distribution>` or `<dataloader-args-for-out-of -distribution>`. Here, an argument for `batch_size` is defined and then either the validation loader, or the training, or the test loader is picked with the corresponding batch size (defined in the `pick_loader`) argument. You can also define a dataset from the list of possible supported datasets (e.g. `cifar10`, `svhn`, `fashion-mnist`, `mnist`, etc.). You can also define an embedding network (e.g. `nvidia_efcicientnet_b0`). Moreover, you may want to perform OOD detection tests between the samples *generated* from a pretrained model (e.g. the current model itself). In which case, you define `dataset` as `dgm-generated` and then add a piece of configuration with the key `dgm_args` that has the following form:

```yaml
dgm_args:
    model_loading_config:
        checkpoint_dir: <directory-of-the-checkpoint-of-the-model>
        config_dir: <directory-of-the-configuration-of-the-model>
    length: <size-of-the-dataset>
    seed: <seed-for-generation>
``` 
Using this, you can sample from a model to obtain a dataset which is then cached for later usage.

**Different visualization schemes**: Under OOD, in place of `<visualization-arguments-and-corresponding-keys>`, you can define multiple configurations to visualize the OOD detection task you are considering.
If you are aiming for speed, then you should bypass the entire visualization process entirely by setting `bypass_visualization=True`. Otherwise, the visualization has three phases:
1. Showing samples from the in- and out-of-distribution datasets. This is intended for image data visualization, however, even for data that has a 1D format, it is reshaped into an appropriate square format and visualized with a grayscale. You can skip this phase by setting `bypass_dataset_visualization=True`.
2. Showing the likelihoods obtained using the in-distribution dataset and out-of-distribution dataset. In this case, the code will iterate over the corresponding samples and visualize a histogram for that matter. A pathology can be easily observed if the likelihoods of the OOD data is larger than the in-distribution data. 
If you don't want to iterate over the entire OOD and in-distribution dataset, you can also define a `histogram_limit` which limits the maximum number of data you would need to calculate for the likelihood histogram.
3. You can also visualize samples drawn from the model itself. This is done by setting `samples_visualization=<0,1,2>`. If set to zero, then this phase is bypassed, if set to `1`, then 9 samples drawn from the model itself (by calling `model.sample(9)`) is drawn and is visualized similar to the first item. If it is set to `2`, it will also call `model.sample(-1)` which in many models produces a special sample (for example, in normalizing flows, this will be the sample obtained by passing zero through the network).

**Single-, batch-, and dataloader-level OOD Detection**: Using the `pick_single` argument, you only perform OOD detection for one particular datapoint. Otherwise, if you set `use_dataloader=False` then a batch of size `pick_count` would be considered. If you set `use_dataloader=True`, then `pick_count` batches would be considered. All of these can be added to the `ood` config as follows:

```yaml
pick_single: <wether-or-not-one-data-is-being-used>
use_dataloader: <whether-or-not-we-are-doing-ood-detection-over-an-antire-dataloader-or-one-batch>
pick_count: <the-number-of-batches-or-datapoints-to-perform-ood-detection>
seed: <for-reproducing-the-same-data>
```

## Training two step models (TODO)

For better generative performance, one can also train two step models where a mapping from feature space to latent space is learned and then the generative modelling task is done on the obtained representation.