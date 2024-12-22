import torch.nn.utils.prune as prune
import torch.nn as nn
import torch

def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')

            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            # ↓毎回マスクを変える処理（復活するノードがある）
            # prune.remove(module, 'weight')


    return model