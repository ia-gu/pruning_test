import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn


import os
import sys
sys.path.append('./')
import argparse
import random
import numpy as np

from src.utils.train import train
from src.utils.get_dataset import build_dataset, build_eval_dataset
from src.eval.evaluate import evaluate
from src.eval.model import CNNModel

def main(args):
    set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(model=args.model, classes=args.num_classes, pretrained=True)
    model = model.to(device)

    train_dataset, nb_classes = build_dataset(args)
    eval_dataset, _ = build_eval_dataset(args)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2, generator=generator)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    train(args, model, train_loader, eval_loader, criterion, device)

    print('Pruning前のモデル:')
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f'\nTotal number of non-zero Conv2d parameters: {total_params:,}')
    evaluate(0, model, criterion, eval_loader, device)

    model = prune_model(model, amount=0.01)

    print('Pruning後のモデル:')
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f'\nTotal number of non-zero Conv2d parameters: {total_params:,}')
    evaluate(0, model, criterion, eval_loader, device)



def prune_model(model, amount=0.5):
    '''
    ResNet18の畳み込み層をLNノルムに基づいてPruning
    Args:
        model: モデル
        amount: 剪定率 (0.0 - 1.0, 例: 0.5は50%)
    '''
    for name, module in model.named_modules():
        # 畳み込み層 (Conv2d) に対してPruningを適用
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            prune.remove(module, 'weight')
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Optional for warmup cosine annealing')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--strategy', type=str, default=None)

    parser.add_argument('--wandb_project', type=str, default='pruning')
    parser.add_argument('--wandb_run', type=str, default='debug')
    parser.add_argument('--wandb_entity', type=str, default='ia-gu')
    parser.add_argument('--output_path', type=str, default='./train_logs')

    return parser.parse_args()

def set_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

if __name__ == '__main__':

    args = parse_args()

    main(args)