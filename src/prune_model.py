import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import os

from src.importance import hessian_pruning, hessian_param_pruning

def prune_model(args, model, criterion, train_loader, amount=0.5):
    # 過去のマスクを適用
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')

    if args.importance == 'Hessian':
        hessian_pruning(model, amount=amount, criterion=criterion, train_loader=train_loader)
    elif args.importance == 'HessianParam':
        hessian_param_pruning(model, amount=amount, criterion=criterion, train_loader=train_loader)

    else:
        pruning_functions = {
            'L1': lambda m: prune.ln_structured(m, name='weight', amount=amount, n=1, dim=0),
        }
        if args.importance not in pruning_functions:
            raise ValueError('Invalid importance type')
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'weight'):
                    pruning_functions[args.importance](module)

    # 保存用のモデルを複製し，マスクを適用して追加モジュールをはがす
    # copy.deepcopyなどでは実装できない
    torch.save(model, args.importance+"temp_model.pth")
    tmp_model = torch.load(args.importance+"temp_model.pth")
    os.remove(args.importance+"temp_model.pth")
    for module in tmp_model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')

    return model, tmp_model