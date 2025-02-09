import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from scipy.sparse.linalg import LinearOperator, eigsh

def hessian_pruning(model, amount, criterion, train_loader):
    # Hessian計算
    hessian_diag_approx = _compute_hessian_diagonal_approx(
        model, train_loader, criterion
    )
    # プルーニング
    return _prune_with_hessian_diag_approx(model, hessian_diag_approx, sparsity=amount)

def hessian_param_pruning(model, amount, criterion, train_loader):
    # Hessian計算
    hessian_diag_approx = _compute_hessian_diagonal_approx(
        model, train_loader, criterion
    )
    # プルーニング
    return _prune_with_hessian_diag_approx_param(model, hessian_diag_approx, sparsity=amount)


def _prune_with_hessian_diag_approx(model, hessian_diag_list, sparsity=0.2):
    diag_scores = []
    for diag_tensor in hessian_diag_list:
        diag_scores.append(diag_tensor.view(-1))
    diag_scores = torch.cat(diag_scores)

    num_params = diag_scores.numel()
    k = int(num_params * (1 - sparsity))
    if k <= 0:
        print("sparsity が高すぎてパラメータを全て刈り取る可能性があります。")
        return model
    threshold = torch.topk(diag_scores, k, largest=True).values.min()
    prune_targets = []
    idx = 0

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_size = param.numel()
                param_diag_score = diag_scores[idx : idx + param_size]
                idx += param_size
                # bias項などはスキップ
                if name != "weight":
                    continue
                # パラメータの対角近似スコアが閾値以上のものを残す
                mask_flat = (param_diag_score >= threshold).float()
                mask = mask_flat.view(param.shape)

                prune_targets.append((module, name, mask))
    for (module, name, mask) in prune_targets:
        prune.custom_from_mask(module, name, mask)

    return model


def _prune_with_hessian_diag_approx_param(model, hessian_diag_list, sparsity=0.2):
    diag_scores = []; param_list = []
    for diag_tensor, param in zip(hessian_diag_list, model.parameters()):
        diag_scores.append(torch.abs(diag_tensor.view(-1))) 
        param_list.append(torch.abs(param.view(-1)))
    adjusted_scores = [h * p for h, p in zip(diag_scores, param_list)]
    diag_scores = torch.cat(adjusted_scores)

    num_params = diag_scores.numel()
    k = int(num_params * (1 - sparsity))
    if k <= 0:
        print("sparsity が高すぎてパラメータを全て刈り取る可能性があります。")
        return model
    threshold = torch.topk(diag_scores, k, largest=True).values.min()
    prune_targets = []
    idx = 0

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_size = param.numel()
                # Hessianとパラメータの要素積
                param_diag_score = diag_scores[idx : idx + param_size]
                idx += param_size
                # bias項などはスキップ
                if name != "weight":
                    continue
                # パラメータの対角近似スコアが閾値以上のものを残す
                mask_flat = (param_diag_score >= threshold).float()
                mask = mask_flat.view(param.shape)

                prune_targets.append((module, name, mask))
    for (module, name, mask) in prune_targets:
        prune.custom_from_mask(module, name, mask)

    return model

def _compute_hessian_diagonal_approx(model, dataloader, loss_fn):
    model.to(torch.device('cuda'))
    model.eval()

    hessian_diag = []
    for param in model.parameters():
        hessian_diag.append(torch.zeros_like(param, device=torch.device('cuda')))

    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(torch.device('cuda')), targets.to(torch.device('cuda'))
            batch_size = inputs.size(0)
            total_samples += batch_size
            # パラメータ更新は行わず，勾配グラフのみを構築
            with torch.enable_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            model.zero_grad()
            loss.backward()

            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    nonzero_mask = (param.data != 0)
                    hessian_diag[i][nonzero_mask] += param.grad.data[nonzero_mask] ** 2

    for i in range(len(hessian_diag)):
        hessian_diag[i] /= float(total_samples)

    return hessian_diag

# HACK torch.prune継承のお試し
class L1L2CombinedPruning(prune.BasePruningMethod):
    def __init__(self, amount):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:
            l1_norm = torch.abs(t)
            l2_norm = torch.square(t)
            importance = l1_norm + l2_norm
            topk = torch.topk(importance.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        return super(L1L2CombinedPruning, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )

def l1l2_combined_pruning(module, name, amount, importance_scores=None):
    L1L2CombinedPruning.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module

def count_nonzero_parameters(module, param_name):
    param = getattr(module, param_name)
    nonzero_count = torch.sum(param != 0).item()
    return nonzero_count

if __name__ == "__main__":
    linear = nn.Linear(10, 5)
    
    # プルーニング前の非ゼロパラメータ数を計算
    original_nonzero_count = count_nonzero_parameters(linear, "weight")
    print(f"プルーニング前の非ゼロパラメータ数: {original_nonzero_count}")
    
    # 重みをL1+L2基準でプルーニング
    l1l2_combined_pruning(linear, "weight", amount=1.0)
    
    # プルーニング後の非ゼロパラメータ数を計算
    pruned_nonzero_count = count_nonzero_parameters(linear, "weight")
    print(f"プルーニング後の非ゼロパラメータ数: {pruned_nonzero_count}")
