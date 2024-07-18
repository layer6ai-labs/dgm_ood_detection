# NOTE: The below file is modified from commit `aeaf5fd` of
#       https://github.com/jrmcornish/cif/blob/master/cif/writer.py
# and then modified again to fit Weights & Biases logging.

import os
import datetime
import json
import sys
import typing as th

import numpy as np
import torch
from dotenv import load_dotenv
# from tensorboardX import SummaryWriter

import wandb

RUN_NAME_SPLIT = '_-_-_'

class Tee:
    """This class allows for redirecting of stdout and stderr"""
    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    # TODO: Should redirect all attrs to primary_file if not found here.
    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        # We get problems with ipdb if we don't do this:
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()

class Writer:
    _STDOUT = sys.stdout
    _STDERR = sys.stderr

    def __init__(
        self, 
        tag_group,
        logdir: th.Optional[str] = None,
        make_subdir: bool = True, 
        type: th.Literal['tensorboard', 'wandb'] = 'tensorboard',
        name: th.Optional[str] = None,
        redirect_streams: bool = False,
        config: th.Optional[dict] = None,
        **kwargs
    ):
        # add model_dir as a prefix to logdir if it is available
        load_dotenv(override=True)
        if os.getenv("MODEL_DIR") is not None:
            _pref = os.getenv("MODEL_DIR")
            logdir = _pref if logdir is None else os.path.join(_pref, logdir)
            
        os.makedirs(logdir, exist_ok=True)
            
        if make_subdir:
            if name is None:
                timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
                logdir = os.path.join(logdir, timestamp)
            else:
                # if name contains RUN_NAME_SPLIT, then we assume that the first part is the run name
                logdir = os.path.join(logdir, name.split(RUN_NAME_SPLIT)[0])
                
            os.makedirs(logdir, exist_ok=True)
        
        if config is not None:
            with open(os.path.join(logdir, 'config.json'), 'w') as json_file:
                json.dump(config, json_file, indent=4)
            
        self.type = type
        if type == 'tensorboard':
            raise NotImplementedError("TensorboardX is not supported anymore. Please use W&B for logging.")
            # TODO: fix tensorboard as well
            from tensorboardX import SummaryWriter
            self._writer = SummaryWriter(logdir=logdir, **kwargs)


            assert logdir == self._writer.logdir
        else:
            self.wandb_tables = {}
            # make all the directories in the logdir
            os.makedirs(logdir, exist_ok=True)
            
            self._writer = wandb.init(name=name, **kwargs)
            
        self.logdir = logdir

        self._tag_group = tag_group
        
        if redirect_streams:
            sys.stdout = Tee(
                primary_file=self._STDOUT,
                secondary_file=open(os.path.join(logdir, "stdout"), "a")
            )

            sys.stderr = Tee(
                primary_file=self._STDERR,
                secondary_file=open(os.path.join(logdir, "stderr"), "a")
            )

    def write_table(
        self,
        name: str,
        data: th.Any,
        columns: th.List[str],
    ):
        if self.type == 'wandb':
            self.wandb_tables[name] = wandb.Table(data=data, columns = columns)
        else:
            raise NotImplementedError("writing table for writers other than W&B is not supported!")
        

    def log_scatterplot(
        self,
        name: str,
        title: str,
        data_table_ref: str,
        x: str,
        y: str,
    ):
        if self.type == 'wandb':
            wandb.log(
                dict(
                    name=wandb.plot.scatter(self.wandb_tables[data_table_ref], x, y, title=title)
                )
            )
        else:
            raise NotImplementedError("writing scatterplots for writers other than W&B is not supported!")
        
    
    def write_scalar(self, tag, scalar_value, global_step=None):
        if self.type == 'tensorboard':
            self._writer.add_scalar(self._tag(tag), scalar_value, global_step=global_step)
        else:
            self._writer.log({self._tag(tag): scalar_value})
    
    def write_image(self, tag, img_tensor, global_step=None):
        if self.type == 'tensorboard':
            self._writer.add_image(self._tag(tag), img_tensor, global_step=global_step)
        else:
            self._writer.log({self._tag(tag): [wandb.Image(img_tensor)]})
            
    def write_figure(self, tag, figure, global_step=None):
        if self.type == 'tensorboard':
            self._writer.add_figure(self._tag(tag), figure, global_step=global_step)
        else:
            self._writer.log({self._tag(tag): [wandb.Image(figure)]})
            
    def write_hparams(self, hparam_dict=None, metric_dict=None):
        if self.type == 'tensorboard':
            self._writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
        else:
            self._writer.config.update(hparam_dict, allow_val_change=True)
            
    def write_json(self, tag, data):
        text = json.dumps(data, indent=4)

        if self.type == 'tensorboard':
            self._writer.add_text(
                self._tag(tag),
                4*" " + text.replace("\n", "\n" + 4*" ") # Indent by 4 to ensure codeblock formatting
            )

        json_path = os.path.join(self.logdir, f"{tag}.json")

        with open(json_path, "w") as f:
            f.write(text)

    def write_textfile(self, tag, text):
        path = os.path.join(self.logdir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write(text)

    def write_numpy(self, tag, arr):
        path = os.path.join(self.logdir, f"{tag}.npy")
        np.save(path, arr)
        print(f"Saved array to {path}")

    def write_checkpoint(self, tag, data):
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        checkpoint_path = self._checkpoint_path(tag)

        tmp_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"{os.path.basename(checkpoint_path)}.tmp"
        )

        torch.save(data, tmp_checkpoint_path)
        # replace is atomic, so we guarantee our checkpoints are always good
        os.replace(tmp_checkpoint_path, checkpoint_path)

    def load_checkpoint(self, tag, device):
        return torch.load(self._checkpoint_path(tag), map_location=device)

    def _checkpoint_path(self, tag):
        return os.path.join(self._checkpoints_dir, f"{tag}.pt")

    @property
    def _checkpoints_dir(self):
        return os.path.join(self.logdir, "checkpoints")

    def _tag(self, tag):
        return f"{self._tag_group}/{tag}"
        