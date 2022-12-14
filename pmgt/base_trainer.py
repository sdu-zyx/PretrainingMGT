"""
Created on 2022/01/11
@author Sangwoo Han
"""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.cuda
import torch.nn as nn
import torch.optim
from attrdict import AttrDict
from logzero import logger
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from optuna import Trial
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Optimizer
from transformers import get_scheduler

from .callbacks import MLFlowExceptionCallback
from .optimizers import DenseSparseAdamW
from .utils import set_seed


def _get_gpu_info(num_gpus: int) -> List[str]:
    return [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(num_gpus)]


def get_optimizer(args: AttrDict) -> Optimizer:
    model: nn.Module = args.model

    no_decay = ["bias", "LayerNorm.weight"]

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.decay,
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
    ]

    if args.optim == "adamw":
        optim = DenseSparseAdamW(param_groups)
    elif args.optim == "sgd":
        optim = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"Optimizer {args.optim} is not supported")

    return optim


def get_run(log_dir: str, run_id: str) -> Run:
    client = MlflowClient(log_dir)
    run = client.get_run(run_id)
    return run


def get_ckpt_path(log_dir: str, run_id: str, load_best: bool = False) -> str:
    run = get_run(log_dir, run_id)
    ckpt_root_dir = os.path.join(log_dir, run.info.experiment_id, run_id, "checkpoints")
    ckpt_path = os.path.join(ckpt_root_dir, "last.ckpt")
    if load_best:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        key = [k for k in ckpt["callbacks"].keys() if k.startswith("ModelCheckpoint")][
            0
        ]
        ckpt_path = ckpt["callbacks"][key]["best_model_path"]

    return ckpt_path


class BaseTrainerModel(pl.LightningModule):
    IGNORE_HPARAMS: List[str] = [
        "model",
        "train_dataset",
        "valid_dataset",
        "test_dataset",
        "inference_dataset",
        "train_dataloader",
        "valid_dataloader",
        "test_dataloader",
        "inference_dataloader",
        "device",
        "run_script",
        "log_dir",
        "experiment_name",
        "data_dir",
        "eval_ckpt_path",
        "tags",
        "is_hptuning",
        "inference_result_path",
        "trial",
        "enable_trial_pruning",
    ]

    def __init__(
        self,
        is_hptuning: bool = False,
        trial: Optional[Trial] = None,
        enable_trial_pruning: bool = False,
        **args: Any,
    ) -> None:
        super().__init__()
        self.args = AttrDict(args)
        self.net = self.args.model
        self.is_hptuning = is_hptuning
        self.trial = trial
        self.enable_trial_pruning = enable_trial_pruning
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.args)
        scheduler = None

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self):
        self.logger.experiment.set_tag(self.logger.run_id, "run_id", self.logger.run_id)
        self.logger.experiment.set_tag(self.logger.run_id, "host", os.uname()[1])

        for k, v in self.args.tags:
            self.logger.experiment.set_tag(self.logger.run_id, k, v)

        logger.info(f"experiment: {self.args.experiment_name}")
        logger.info(f"run_name: {self.args.run_name}")
        logger.info(f"run_id: {self.logger.run_id}")

        # if "run_script" in self.args:
        #     self.logger.experiment.log_artifact(
        #         self.logger.run_id, self.args.run_script, "scripts"
        #     )
        if "run_script" in self.args:
            self.logger.experiment.log_artifact(
                self.logger.run_id, "scripts"
            )

        logger.info(f"num_gpus: {self.args.num_gpus}, no_cuda:{self.args.no_cuda}")
        if self.args.num_gpus >= 1 and not self.args.no_cuda:
            gpu_info = _get_gpu_info(self.args.num_gpus)
            self.logger.experiment.set_tag(
                self.logger.run_id, "GPU info", ", ".join(gpu_info)
            )

    def should_prune(self, value):
        if self.trial is not None and self.enable_trial_pruning:
            self.trial.report(value, self.global_step)
            if self.trial.should_prune():
                self.logger.experiment.set_tag(self.logger.run_id, "pruned", True)
                raise optuna.TrialPruned()


def init_run(args: AttrDict) -> None:
    if args.seed is not None:
        logger.info(f"seed: {args.seed}")
        set_seed(args.seed)

    args.device = torch.device("cpu" if args.no_cuda else "cuda")
    # args.num_gpus = torch.cuda.device_count()
    args.num_gpus = 1

def check_args(
    args: AttrDict,
    early_criterion: List[str],
    model_name: List[str],
    dataset_name: List[str],
) -> None:
    if args.mode in ["eval", "inference"]:
        assert args.run_id is not None, f"run_id must be provided in mode {args.mode}"
    print(args.early_criterion, early_criterion)
    assert (
        args.early_criterion in early_criterion
    ), f"early_criterion must be one of {early_criterion}"

    assert args.model_name in model_name, f"model_name must be one of {model_name}"

    assert (
        args.dataset_name in dataset_name
    ), f"dataset_name must be one of {dataset_name}"

    assert type(args.valid_size) in [float, int], "valid size must be int or float"


def init_dataloader(
    args: AttrDict, get_dataset: Callable, get_dataloader: Callable
) -> None:
    logger.info(f"Dataset: {args.dataset_name}")
    train_dataset, valid_dataset, test_dataset = get_dataset(args)
    args.train_dataset = train_dataset
    args.valid_dataset = valid_dataset
    args.test_dataset = test_dataset

    logger.info(f"Prepare Dataloader")
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(
        args,
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    args.train_dataloader = train_dataloader
    args.valid_dataloader = valid_dataloader
    args.test_dataloader = test_dataloader


def init_model(args: AttrDict, get_model: Callable) -> None:
    logger.info(f"Model: {args.model_name}")

    model = get_model(args)

    if args.num_gpus > 1 and not args.no_cuda:
        logger.info(f"Multi-GPU mode: {args.num_gpus} GPUs")
    elif not args.no_cuda:
        logger.info("single-GPU mode")
    else:
        logger.info("CPU mode")

    if args.num_gpus >= 1 and not args.no_cuda:
        gpu_info = _get_gpu_info(args.num_gpus)
        logger.info("GPU info - " + " ".join(gpu_info))

    args.model = model


def train(
    args: AttrDict,
    TrainerModel: Type[BaseTrainerModel],
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
    **trainer_model_args,
) -> Tuple[float, pl.Trainer]:
    if args.run_name is None:
        run_name = f"{args.model_name}_{args.dataset_name}"
    else:
        run_name = args.run_name

    logger.info(f"run_name + {args.run_name}")

    mlf_logger = pl_loggers.MLFlowLogger(
        experiment_name=args.experiment_name, run_name=run_name, save_dir=args.log_dir
    )

    monitor = (
        "loss/val" if args.early_criterion == "loss" else f"val/{args.early_criterion}"
    )
    mode = "min" if args.early_criterion == "loss" else "max"

    early_stopping_callback = EarlyStopping(
        monitor=monitor, patience=args.early, mode=mode
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename=f"epoch={{epoch:02d}}-{monitor.split('/')[-1]}={{{monitor}:.4f}}",
        mode=mode,
        save_top_k=1,
        auto_insert_metric_name=False,
        save_last=True,
    )
    mlflow_exception_callback = MLFlowExceptionCallback()

    logger.info(f"monitor + {monitor}")
    logger.info(f"monitor mode + {mode}")

    trainer_model = TrainerModel(
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
        **trainer_model_args,
        **args,
    )

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        gpus=args.num_gpus,
        precision=16 if args.mp_enabled else 32,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_max_norm,
        accumulate_grad_batches=args.accumulation_step,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
            mlflow_exception_callback,
        ],
        logger=mlf_logger,
    )
    logger.info(f"config pl trainer over")

    ckpt_path = get_ckpt_path(args.log_dir, args.run_id, False) if args.run_id else None
    logger.info(f"log_dir + {args.log_dir}")
    logger.info(f"run_id + {args.run_id}")
    logger.info(f"ckpt_path + {ckpt_path}")

    try:
        trainer.fit(
            trainer_model,
            args.train_dataloader,
            args.valid_dataloader,
            ckpt_path=ckpt_path,
        )
    except optuna.TrialPruned:
        pass

    args.eval_ckpt_path = checkpoint_callback.best_model_path

    best_score = checkpoint_callback.best_model_score
    best_score = best_score.item() if best_score else 0

    return best_score, trainer


def test(
    args: AttrDict,
    TrainerModel: Type[BaseTrainerModel],
    metrics: List[str],
    trainer: Optional[pl.Trainer] = None,
    is_hptuning: bool = False,
    **trainer_model_args,
) -> Dict[str, float]:
    if args.mode == "eval":
        ckpt_path = get_ckpt_path(args.log_dir, args.run_id, True)
    else:
        ckpt_path = args.eval_ckpt_path

    trainer_model = TrainerModel(is_hptuning=is_hptuning, **trainer_model_args, **args)

    trainer = trainer or pl.Trainer(
        gpus=args.num_gpus,
        precision=16 if args.mp_enabled else 32,
        enable_model_summary=False,
        logger=False,
    )

    results = trainer.test(
        trainer_model,
        args.valid_dataloader if is_hptuning else args.test_dataloader,
        ckpt_path=ckpt_path or None,
        verbose=False,
    )

    if results is not None:
        results = results[0]

        msg = "\n" + "\n".join([f"{m}: {results['test/' + m]:.4f}" for m in metrics])
        logger.info(msg)

    return results or {}


def inference(
    args: AttrDict,
    TrainerModel: Type[BaseTrainerModel],
    **trainer_model_args,
) -> np.ndarray:
    assert args.mode == "inference", "mode must be inference"

    ckpt_path = get_ckpt_path(args.log_dir, args.run_id, True)

    trainer_model = TrainerModel(**trainer_model_args, **args)

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        precision=16 if args.mp_enabled else 32,
        enable_model_summary=False,
        logger=False,
    )
    print("start predict")
    predictions = trainer.predict(
        trainer_model, args.inference_dataloader, ckpt_path=ckpt_path
    )
    print(args.inference_dataloader, predictions)
    predictions = np.concatenate(predictions)

    if args.inference_result_path is not None:
        np.save(args.inference_result_path, predictions)
        logger.info(f"Save result: {args.inference_result_path}")

    return predictions
