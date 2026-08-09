"""Block-level model evaluation."""
from collections import defaultdict
import numpy as np
import torch
from .metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, num_classes=5, class_names=None, mc_samples=1):
    """Average subepoch logits by block ID, then calculate paper metrics."""
    model.eval(); logits_all=[]; labels_all=[]; blocks_all=[]; total_loss=0.0
    if mc_samples > 1:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.train()
    for x,y,blocks in loader:
        x=x.to(device); y=y.to(device); model.reset_state(len(x),device)
        logits=sum(model(x,sequence_start=True)["logits"] for _ in range(mc_samples))/mc_samples
        if criterion is not None: total_loss += float(criterion(logits,y))*len(x)
        logits_all.extend(logits.cpu().numpy()); labels_all.extend(y.cpu().numpy()); blocks_all.extend(blocks.numpy())
    grouped=defaultdict(list); block_labels={}
    for logits,label,block in zip(logits_all,labels_all,blocks_all): grouped[int(block)].append(logits); block_labels[int(block)]=int(label)
    truth=[]; predictions=[]; probabilities=[]
    for block in sorted(grouped):
        mean_logits=np.mean(grouped[block],axis=0); exp=np.exp(mean_logits-np.max(mean_logits)); prob=exp/exp.sum()
        truth.append(block_labels[block]); predictions.append(int(np.argmax(mean_logits))); probabilities.append(prob)
    probs=np.vstack(probabilities) if probabilities else np.zeros((0,num_classes))
    metrics=compute_metrics(truth,predictions,probs,num_classes,class_names)
    metrics["loss"]=total_loss/len(loader.dataset) if criterion is not None else 0.0
    return metrics
