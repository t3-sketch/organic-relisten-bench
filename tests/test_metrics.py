import numpy as np
import polars as pl

from relisten.metrics import evaluate_probabilities, paired_user_auc_difference, user_auc_table


def test_auc_uses_average_ranks_for_ties_and_excludes_single_class_users():
    table = user_auc_table([1,1,2,2,3,3],[0,1,0,1,1,1],[.5,.5,.1,.9,.2,.3])
    assert table["uid"].to_list()==[1,2]
    assert table["auc"].to_list()==[.5,1.0]


def test_equal_user_weight_instead_of_pair_weight():
    result, users = evaluate_probabilities([1,1,2,2,2,2],[0,1,0,0,1,1],[.9,.1,.1,.2,.8,.9],n_bootstrap=100)
    assert result["user_auc"]==.5
    assert result["auc_users"]==2
    assert result["global_auc"]!=.5
    same = paired_user_auc_difference(users,users,n_bootstrap=100)
    assert same["mean_auc_difference"]==0
    assert same["difference_ci_low"]==same["difference_ci_high"]==0


def test_perfect_and_constant_probabilities():
    uid,y = np.repeat([1,2],2),np.tile([0,1],2)
    perfect,_ = evaluate_probabilities(uid,y,y.astype(float),n_bootstrap=100)
    constant,_ = evaluate_probabilities(uid,y,np.full(4,.5),n_bootstrap=100)
    assert perfect["user_auc"]==perfect["pr_auc"]==1
    assert perfect["brier"]==0
    assert constant["user_auc"]==.5
    assert constant["brier"]==.25


def test_paired_bootstrap_is_invariant_to_input_user_order():
    left=pl.DataFrame({'uid':[3,1,2,4],'auc':[.5,.8,.4,.9]})
    right=pl.DataFrame({'uid':[1,2,3,4],'auc':[.6,.7,.5,.7]})
    assert paired_user_auc_difference(left,right)==paired_user_auc_difference(left.reverse(),right.reverse())
