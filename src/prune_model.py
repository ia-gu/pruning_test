import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import os

from src.importance import hessian_pruning, hessian_param_pruning, jacobian_pruning, jacobian_param_pruning, param_pruning

def prune_model(args, model, criterion, train_loader, sparsity=0.5):
    # 過去のマスクを適用
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')

    if args.importance == 'Hessian':
        hessian_pruning(model, sparsity=sparsity, criterion=criterion, train_loader=train_loader, args=args)
    elif args.importance == 'HessianParam':
        hessian_param_pruning(model, sparsity=sparsity, criterion=criterion, train_loader=train_loader, args=args)
    elif args.importance == 'Jacobian':
        jacobian_pruning(model, sparsity=sparsity, train_loader=train_loader, args=args)
    elif args.importance == 'JacobianParam':
        jacobian_param_pruning(model, sparsity=sparsity, train_loader=train_loader, args=args)
    elif args.importance == 'Magnitude':
        param_pruning(model, sparsity, args=args)

    else:
        pruning_functions = {
            'L1': lambda m: prune.ln_structured(m, name='weight', amount=sparsity, n=1, dim=0),
        }
        if args.importance not in pruning_functions:
            raise ValueError('Invalid importance type')
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'weight'):
                    pruning_functions[args.importance](module)

    # 保存用のモデルを複製し，マスクを適用して追加モジュールをはがす
    # copy.deepcopyなどでは実装できない
    torch.save(model, os.path.join(args.output_path, 'ckpt', 'checkpoint.pth'))
    tmp_model = torch.load(os.path.join(args.output_path, 'ckpt', 'checkpoint.pth'))
    os.remove(os.path.join(args.output_path, 'ckpt', 'checkpoint.pth'))
    for module in tmp_model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')

    return model, tmp_model