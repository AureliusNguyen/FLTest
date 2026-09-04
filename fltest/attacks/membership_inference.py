"""Membership inference: decide whether a sample was in a client's training data.

This is the canonical privacy attack the proposal cites (Shokri et al., IEEE S&P 2017) and
the one Pitfall-1 says evaluations skip. It models an honest-but-curious server that sees
the global model each round and wants to know whether a particular record trained it.

The scoring rule is the loss threshold of Yeom et al. (CSF 2018): a model tends to assign
lower loss to data it trained on, so per-sample loss separates members from non-members
without any shadow model. Members are the target client's own training data and
non-members are the held-out test set, and the reported AUC is the probability that a
random member scores as more member-like than a random non-member. 0.5 means no leakage
and 1.0 means the two are perfectly separable.

Unlike gradient inversion, this reads only losses, so it applies to text as readily as to
images.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack
from fltest.data.models import LOSS_FUNCTIONS, forward_batch, get_model
from fltest.data.utils import load_ndarrays_into


def roc_auc(member_scores: np.ndarray, non_member_scores: np.ndarray) -> float:
    """AUC by the rank formulation, so no extra dependency is needed.

    Equivalent to the Mann-Whitney U statistic normalised by the number of pairs. Ties
    take the average rank, which is what keeps a model that scores everything identically
    at 0.5 rather than at an accidental 0 or 1.
    """
    n_pos, n_neg = len(member_scores), len(non_member_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = np.concatenate([member_scores, non_member_scores])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    ordered = scores[order]
    start = 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + 1 + stop + 1) / 2.0
        start = stop + 1

    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@register_attack("membership_inference")
class MembershipInferenceAttack(ThreatModelBaseClass):
    """Loss-threshold membership inference against the global model, scored each round."""

    HOOKS = ("on_data_distribute", "after_round")

    def __init__(
        self,
        target_client: int = 0,
        max_samples: int = 512,
        **params,
    ):
        super().__init__(**params)
        self.target_client = target_client
        self.max_samples = max_samples
        self._client_loaders = None

    def on_data_distribute(self, ctx: HookContext) -> None:
        # The victim's training data is the member set, so hold on to it while it is here.
        if ctx.dist_dict is not None:
            self._client_loaders = dict(ctx.dist_dict)

    @torch.no_grad()
    def _per_sample_losses(self, model, loader, criterion, device: str) -> np.ndarray:
        losses: List[float] = []
        for batch in loader:
            labels = batch["label"].to(device)
            logits = forward_batch(model, batch, device)
            losses.extend(criterion(logits, labels).detach().cpu().tolist())
            if len(losses) >= self.max_samples:
                break
        return np.asarray(losses[: self.max_samples], dtype=float)

    @torch.no_grad()
    def after_round(self, ctx: HookContext) -> None:
        loaders, spec = self._client_loaders, ctx.cfg
        if not loaders or spec is None or ctx.test_data is None:
            return
        member_loader = loaders.get(self.target_client)
        if member_loader is None or ctx.global_state is None:
            return

        model = get_model(
            spec.model_name, spec.model_cache_path, channels=spec.channels,
            num_classes=spec.num_classes, deterministic=False,
        ).to(spec.device)
        load_ndarrays_into(model, ctx.global_state)
        model.eval()

        criterion = LOSS_FUNCTIONS[spec.loss_fn](reduction="none")
        member = self._per_sample_losses(model, member_loader, criterion, spec.device)
        non_member = self._per_sample_losses(model, ctx.test_data, criterion, spec.device)
        if len(member) == 0 or len(non_member) == 0:
            return

        # A member is expected to have the lower loss, so negate to make "more member-like"
        # the larger score.
        auc = roc_auc(-member, -non_member)
        ctx.record(
            membership_inference_auc=auc,
            membership_loss_gap=float(non_member.mean() - member.mean()),
        )
