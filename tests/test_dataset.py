import numpy as np
import torch
from sleep_staging.data.dataset import SleepDataset


def test_dataset_loading_and_channel_selection(tmp_path):
    X=np.zeros((4,3,500,2),dtype=np.float32); X[...,1]=7; y=np.array([0,1,2,4]); groups=np.array([10,10,11,11])
    paths=[]
    for name,value in (("X.npy",X),("y.npy",y),("g.npy",groups)):
        path=tmp_path/name; np.save(path,value); paths.append(path)
    dataset=SleepDataset(*paths,channel_indices=(1,)); signal,label,group=dataset[3]
    assert len(dataset)==4 and signal.shape==(3,500,1) and torch.all(signal==7)
    assert label.dtype==torch.long and 0 <= label < 5 and group.item()==11
