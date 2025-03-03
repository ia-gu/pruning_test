import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def hessian_pruning(model, sparsity, criterion, train_loader, args=None):
    # Hessian計算
    hessian_diag_approx = _compute_hessian_diagonal_approx(
        model, train_loader, criterion
    )
    # プルーニング
    return _prune_with_hessian_diag_approx(model, hessian_diag_approx, sparsity=sparsity, args=args)

def hessian_param_pruning(model, sparsity, criterion, train_loader, args=None):
    # Hessian計算
    hessian_diag_approx = _compute_hessian_diagonal_approx(
        model, train_loader, criterion
    )
    # プルーニング
    return _prune_with_hessian_diag_approx_param(model, hessian_diag_approx, sparsity=sparsity, args=args)

def jacobian_pruning(model, sparsity, train_loader, args=None):
    # Jacobian計算
    jacobian = _compute_jacobian(model, train_loader)
    # プルーニング
    return _prune_with_jacobian(model, jacobian, sparsity=sparsity, args=args)

def jacobian_param_pruning(model, sparsity, train_loader, args=None):
    # Jacobian計算
    jacobian = _compute_jacobian(model, train_loader)
    # プルーニング
    return _prune_with_jacobian_param(model, jacobian, sparsity=sparsity, args=args)

def param_pruning(model, sparsity, args=None):
    return _prune_with_param(model, sparsity=sparsity, args=args)

def _prune_with_hessian_diag_approx(model, hessian_diag_list, sparsity, args=None):
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


def _prune_with_hessian_diag_approx_param(model, hessian_diag_list, sparsity, args=None):
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

def _prune_with_jacobian(model, jacobian_list, sparsity, args=None):
    # 各テンソルを1次元に変換して連結
    jacobian_scores = []
    for jacobian_tensor in jacobian_list:
        jacobian_scores.append(jacobian_tensor.view(-1))
    jacobian_scores = torch.cat(jacobian_scores)

    num_params = jacobian_scores.numel()
    k = int(num_params * (1 - sparsity))
    if k <= 0:
        print("sparsity が高すぎてパラメータを全て刈り取る可能性があります。")
        return model

    # k番目の大きさを閾値とする
    threshold = torch.topk(jacobian_scores, k, largest=True).values.min()
    prune_targets = []
    idx = 0

    # 各パラメータごとにマスクを作成（weight 以外はスキップ）
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_size = param.numel()
                param_jacobian_score = jacobian_scores[idx: idx + param_size]
                idx += param_size
                # bias などはスキップ
                if name != "weight":
                    continue
                # ヤコビアンスコアが閾値以上なら1（残す），未満なら0（枝刈り）
                mask_flat = (param_jacobian_score >= threshold).float()
                mask = mask_flat.view(param.shape)
                prune_targets.append((module, name, mask))
    
    # prune.custom_from_mask を用いて枝刈りを適用
    for (module, name, mask) in prune_targets:
        prune.custom_from_mask(module, name, mask)

    return model

def _prune_with_jacobian_param(model, jacobian_list, sparsity, args=None):
    # 各テンソルを1次元に変換して連結
    jacobian_scores = []; param_list = []
    for jacobian_tensor, param in zip(jacobian_list, model.parameters()):
        jacobian_scores.append(torch.abs(jacobian_tensor.view(-1)))
        param_list.append(torch.abs(param.view(-1)))
    adjusted_scores = [j * p for j, p in zip(jacobian_scores, param_list)]
    jacobian_scores = torch.cat(adjusted_scores)

    num_params = jacobian_scores.numel()
    k = int(num_params * (1 - sparsity))
    if k <= 0:
        print("sparsity が高すぎてパラメータを全て刈り取る可能性があります。")
        return model

    # k番目の大きさを閾値とする
    threshold = torch.topk(jacobian_scores, k, largest=True).values.min()
    prune_targets = []
    idx = 0

    # 各パラメータごとにマスクを作成（weight 以外はスキップ）
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_size = param.numel()
                param_jacobian_score = jacobian_scores[idx: idx + param_size]
                idx += param_size
                # bias などはスキップ
                if name != "weight":
                    continue
                # ヤコビアンスコアが閾値以上なら1（残す），未満なら0（枝刈り）
                mask_flat = (param_jacobian_score >= threshold).float()
                mask = mask_flat.view(param.shape)
                prune_targets.append((module, name, mask))
    
    # prune.custom_from_mask を用いて枝刈りを適用
    for (module, name, mask) in prune_targets:
        prune.custom_from_mask(module, name, mask)

    return model

def _prune_with_param(model, sparsity, args=None):
    param_list = torch.cat([param.view(-1) for param in model.parameters()])

    num_params = param_list.numel()
    k = int(num_params * (1 - sparsity))
    if k <= 0:
        print("sparsity が高すぎてパラメータを全て刈り取る可能性があります。")
        return model

    threshold = torch.topk(param_list, k, largest=True).values.min()
    prune_targets = []
    idx = 0

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_size = param.numel()
                param_jacobian_score = param_list[idx: idx + param_size]
                idx += param_size
                if name != "weight":
                    continue
                mask_flat = (param_jacobian_score >= threshold).float()
                mask = mask_flat.view(param.shape)
                prune_targets.append((module, name, mask))

                if args.verbose:
                    retained = torch.sum(mask).item()
                    total = mask.numel()
                    retention_rate = retained / total * 100
                    pruned = total - retained
                    with open(os.path.join(args.args.output_path, 'ckpt', 'logs.txt'), 'a+') as f:
                        print(f"モジュール: {module.__class__.__name__}, パラメータ: {name}", file=f)
                        print(f"  形状: {mask.shape}, 総パラメータ数: {total}", file=f)
                        print(f"  残したパラメータ: {retained} ({retention_rate:.2f}%)", file=f)
                        print(f"  プルーニングしたパラメータ: {pruned} ({100-retention_rate:.2f}%)", file=f)

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

def _compute_jacobian(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    params = list(model.parameters())
    jacobian_approx = [torch.zeros_like(p, device=device) for p in params]
    total_elements = 0

    # バッチ処理を効率化
    with torch.enable_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # バッチ全体で一度に計算
            outputs = model(inputs)
            output_dim = outputs.numel() // batch_size
            total_elements += batch_size * output_dim

            # 出力の次元に応じて処理を分岐
            if output_dim > 1:
                # reshape して効率的に処理
                outputs_flat = outputs.view(batch_size, -1)
                for j in range(outputs_flat.size(1)):
                    # バッチ全体の勾配を一度に計算
                    grad_outputs = torch.zeros_like(outputs_flat)
                    grad_outputs[:, j] = 1.0
                    grads = torch.autograd.grad(
                        outputs_flat, params,
                        grad_outputs=grad_outputs,
                        retain_graph=(j < outputs_flat.size(1) - 1),
                        allow_unused=True
                    )
                    for idx, grad in enumerate(grads):
                        if grad is not None:
                            # バッチ方向に集約
                            jacobian_approx[idx] += grad.pow(2).sum(0)
            else:
                # スカラー出力の場合
                grad_outputs = torch.ones_like(outputs)
                grads = torch.autograd.grad(
                    outputs, params,
                    grad_outputs=grad_outputs,
                    allow_unused=True
                )
                for idx, grad in enumerate(grads):
                    if grad is not None:
                        jacobian_approx[idx] += grad.pow(2).sum(0)

            # メモリ効率化のためにキャッシュをクリア
            torch.cuda.empty_cache()

    # 正規化
    for idx in range(len(jacobian_approx)):
        jacobian_approx[idx] /= total_elements

    return jacobian_approx

# HACK torch.prune継承のお試し
class L1L2CombinedPruning(prune.BasePruningMethod):
    def __init__(self, sparsity):
        prune._validate_pruning_sparsity_init(sparsity)
        self.sparsity = sparsity

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.sparsity, tensor_size)
        prune._validate_pruning_sparsity(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:
            l1_norm = torch.abs(t)
            l2_norm = torch.square(t)
            importance = l1_norm + l2_norm
            topk = torch.topk(importance.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, sparsity, importance_scores=None):
        return super(L1L2CombinedPruning, cls).apply(
            module, name, sparsity=sparsity, importance_scores=importance_scores
        )

def l1l2_combined_pruning(module, name, sparsity, importance_scores=None):
    L1L2CombinedPruning.apply(
        module, name, sparsity=sparsity, importance_scores=importance_scores
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
    l1l2_combined_pruning(linear, "weight", sparsity=1.0)
    
    # プルーニング後の非ゼロパラメータ数を計算
    pruned_nonzero_count = count_nonzero_parameters(linear, "weight")
    print(f"プルーニング後の非ゼロパラメータ数: {pruned_nonzero_count}")
