# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# modified by Stella Sangyoon Bae

from fairseq.dataclass.configs import FairseqDataclass

#import numpy as np
import math
import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from sklearn.metrics import r2_score

# negative R2 -> 그래야 R2의 절댓값이 젤 높은 걸 찾아냄.


@register_criterion("negative_r2", dataclass=FairseqDataclass)
class GraphPredictionR2(FairseqCriterion):
    """
    Implementation for the rmse loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])

        maybe_loss = (r2_score(logits.detach().cpu().numpy(), targets[: logits.size(0)].detach().cpu().numpy())) * (-1)
        loss = torch.tensor(maybe_loss, requires_grad=True)
        
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("negative_r2_with_flag", dataclass=FairseqDataclass)
class GraphPredictionR2WithFlag(GraphPredictionR2):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def perturb_forward(self, model, sample, perturb, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        maybe_loss = (r2_score(logits.detach().cpu().numpy(), targets[: logits.size(0)].detach().cpu().numpy())) * (-1)
        loss = torch.tensor(maybe_loss, requires_grad=True)
        
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output
