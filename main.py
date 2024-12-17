import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from evaluate import evaluate
from model import CNNModel

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(model='ResNet50', classes=1000, pretrained=True)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(root=os.path.join("/data01", "imagenet", "val_256"), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    print("Pruning前のモデル:")
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f"\nTotal number of non-zero Conv2d parameters: {total_params:,}")
    evaluate(0, model, criterion, test_loader, device)

    model = prune_model(model, amount=0.7)

    print("Pruning後のモデル:")
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f"\nTotal number of non-zero Conv2d parameters: {total_params:,}")
    evaluate(0, model, criterion, test_loader, device)



def prune_model(model, amount=0.5):
    """
    ResNet18の畳み込み層をLNノルムに基づいてPruning
    Args:
        model: モデル
        amount: 剪定率 (0.0 - 1.0, 例: 0.5は50%)
    """
    for name, module in model.named_modules():
        # 畳み込み層 (Conv2d) に対してPruningを適用
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            prune.remove(module, "weight")
    return model

if __name__ == "__main__":
    main()