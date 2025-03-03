import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import os
import sys
sys.path.append('./')
import yaml
import random
import numpy as np
import wandb
import argparse
from datetime import datetime

from src.train import train
from src.train_iterative import train_iterative
from src.get_dataset import build_dataset, build_eval_dataset
from src.model import CNNModel


def main(args):
    set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, nb_classes = build_dataset(args)
    if nb_classes != args.num_classes:
        raise ValueError('Number of classes in dataset and num_classes should be the same')
    eval_dataset, _ = build_eval_dataset(args)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=2, generator=generator, worker_init_fn=worker_init_fn)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    model = CNNModel(model=args.model, classes=args.num_classes, pretrained=False)
    if args.weight_path:
        model.load_state_dict(torch.load(args.weight_path))
    model = model.to(device)
    if args.prune_mode == 'epoch':
        print('----------   Prune Per Epochs   ----------')
        train(args, model, train_loader, eval_loader, criterion, device)
    elif args.prune_mode == 'iteration':
        print('----------   Prune Per Iterations   ----------')
        train_iterative(args, model, train_loader, eval_loader, criterion, device)


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='ImageNet')

    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--train_method', type=str, default='None')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Optimal for warmup pruning schedule')
    parser.add_argument('--prune_mode', type=str, default='epoch')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=True)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cosine_warmup_epochs', type=int, default=0, help='Optional for warmup cosine annealing')
    parser.add_argument('--importance', type=str, default='None', help='None: なし, L1: パラメータ, Hessian: ヘッシアン, HessianParam: パラメータ×ヘッシアン')

    parser.add_argument('--wandb_project', type=str, default='pruning')
    parser.add_argument('--wandb_run', type=str, default='debug')
    parser.add_argument('--wandb_entity', type=str, default='ia-gu')
    parser.add_argument('--output_path', type=str, default='./logs')

    return parser.parse_args()

def set_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':

    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    args.output_path = os.path.join(args.output_path, args.wandb_run, args.model, args.dataset, args.importance, str(args.lr), str(args.seed), str(args.pruning_ratio), timestamp)
    os.makedirs(os.path.join(args.output_path, 'ckpt'), exist_ok=True)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run+'_'+args.model+'_'+args.dataset+'_'+args.importance+'_'+str(args.lr)+'_'+str(args.seed)+'_'+str(args.pruning_ratio), config=vars(args))
    # output args to yaml file
    with open(os.path.join(args.output_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    main(args)