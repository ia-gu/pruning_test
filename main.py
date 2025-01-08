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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2, generator=generator)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    model = CNNModel(model=args.model, classes=args.num_classes, pretrained=True)
    model = model.to(device)

    train(args, model, train_loader, eval_loader, criterion, device)




def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pruning_ratio', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Optional for warmup cosine annealing')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--importance', type=str, default='L1')

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