import numpy as np
from sleep_staging.data.splits import generate_folds


def test_splits_are_reproducible_and_group_disjoint():
    groups=np.repeat(np.arange(10),4); labels=np.tile([0,1,0,1],10)
    first=list(generate_folds(labels,groups)); second=list(generate_folds(labels,groups))
    for (train,val),(train2,val2) in zip(first,second):
        assert not set(groups[train]) & set(groups[val]); np.testing.assert_array_equal(train,train2); np.testing.assert_array_equal(val,val2)
