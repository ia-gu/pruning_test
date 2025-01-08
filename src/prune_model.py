import torch.nn.utils.prune as prune

from src.importance import hessian_pruning

def prune_model(args, model, criterion, train_loader, amount=0.5):
    # 過去のマスクを適用
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')

    if args.importance == 'Hessian':
        hessian_pruning(model, amount=amount, criterion=criterion, train_loader=train_loader)

    else:
        pruning_functions = {
            'L1': lambda m: prune.ln_structured(m, name='weight', amount=amount, n=1, dim=0),
        }
        if args.importance not in pruning_functions:
            raise ValueError('Invalid importance type')
        pruning_functions[args.importance](module)

    # ↓モデル保存用
    tmp_model = model
    for module in tmp_model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')


    return model, tmp_model