"""Membership inference scoring, and the ranking statistic behind it."""

import numpy as np
import pytest

from fltest.attacks.membership_inference import MembershipInferenceAttack, roc_auc


def test_auc_is_one_when_members_score_higher():
    assert roc_auc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 1.0


def test_auc_is_zero_when_the_ordering_is_reversed():
    assert roc_auc(np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0])) == 0.0


def test_all_ties_score_exactly_chance():
    """A model that scores everything alike leaks nothing, so ties must average to 0.5."""
    assert roc_auc(np.ones(4), np.ones(4)) == 0.5


def test_partial_overlap_lands_between():
    auc = roc_auc(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.5, 2.5]))
    assert 0.5 < auc < 1.0


def test_empty_side_is_not_a_number_rather_than_a_crash():
    assert np.isnan(roc_auc(np.array([]), np.array([1.0, 2.0])))


def test_attack_is_inert_without_the_data_it_needs():
    """Backends that never populate dist_dict must not break the run."""
    from fltest.core.hook_context import HookContext

    attack = MembershipInferenceAttack()
    ctx = HookContext()
    attack.after_round(ctx)          # no client loaders captured yet
    assert ctx.metrics == {}


@pytest.mark.parametrize("bad", [np.array([]), np.zeros(0)])
def test_auc_handles_degenerate_input(bad):
    assert np.isnan(roc_auc(bad, np.array([1.0])))
