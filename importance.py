import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class L1L2CombinedPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

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
