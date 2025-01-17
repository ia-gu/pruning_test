import torch
import torch.backends.cudnn as cudnn
from timm.utils import accuracy


import os
import sys
sys.path.append('./')
import random
import numpy as np
import wandb
import argparse
from datetime import datetime

from src.get_dataset import build_test_dataset
from src.model import CNNModel

def main(args):
    set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(model=args.model, classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.weight_path, args.epoch+'.pth')))

    model = model.to(device)
    test_datasets = build_test_dataset(args)

    dataset_types = ['clean', 'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 
                     'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
    for i in range(len(test_datasets)):
        if not len(test_datasets) == len(dataset_types):
            raise ValueError('Datasets and dataset_types should have the same length')
        if test_datasets[i] == None:
            continue
        test_data_loader = torch.utils.data.DataLoader(
            test_datasets[i],
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
        evaluate_robustness(test_data_loader, model, device, args, dataset_types[i])

# クラスごとの正解率を出す
class ClassAccuracyMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.correct_per_class = torch.zeros(num_classes)
        self.total_per_class = torch.zeros(num_classes)

    def update(self, preds, targets):
        for p, t in zip(preds, targets):
            if p == t:
                self.correct_per_class[t] += 1
            self.total_per_class[t] += 1

    def get_accuracy(self):
        per_class_acc = 100.0 * (self.correct_per_class / (self.total_per_class + 1e-6))
        mean_class_acc = per_class_acc.mean().item()
        return per_class_acc.tolist(), mean_class_acc


@torch.no_grad()
def evaluate_robustness(data_loader, model, device, args, dataset_type):
    criterion = torch.nn.CrossEntropyLoss()

    # 評価モードに切り替え
    model.eval()

    # クラス名リスト (元コードと同様に定義)
    if args.dataset == 'CIFAR10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'CIFAR100':
        classes = [i for i in range(100)]
    elif args.dataset == 'TINY':
        classes = [i for i in range(200)]
    else:
        classes = []

    class_acc_meter = ClassAccuracyMeter(len(classes))

    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_samples = 0

    print_freq = 50

    for batch_idx, (images, target) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, predictions = output.max(dim=1)
        class_acc_meter.update(predictions, target)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc1 += acc1.item() * batch_size
        total_acc5 += acc5.item() * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % print_freq == 0:
            print(f"[{batch_idx+1:>3}/{len(data_loader)}] " f"Loss: {loss.item():.4f}, " f"Acc@1: {acc1.item():.2f}, " f"Acc@5: {acc5.item():.2f}")

    avg_loss = total_loss / total_samples
    avg_acc1 = (total_acc1 / total_samples)
    avg_acc5 = (total_acc5 / total_samples)
    per_class_acc, mean_class_acc = class_acc_meter.get_accuracy()


########################################以下は結果表示########################################
    print(f'{args.dataset}, {dataset_type}')
    print(f'* Acc@1 {avg_acc1:.3f}  Acc@5 {avg_acc5:.3f}  Loss {avg_loss:.3f}')

    os.makedirs(os.path.join(args.output_path, '../test', args.epoch), exist_ok=True)
    with open(os.path.join(args.output_path, '../test', args.epoch, 'total_result.txt'), "a") as f:
        f.write(f'{args.dataset}, {dataset_type}')
        f.write(f'* Acc@1 {avg_acc1:.3f}  Acc@5 {avg_acc5:.3f}  Loss {avg_loss:.3f}\n')

    with open(os.path.join(args.output_path, '../test', args.epoch, 'class_wise_result.txt'), "a") as f:
        f.write(f'{args.dataset}, {dataset_type}')
        for i, (cls, acc) in enumerate(zip(classes, per_class_acc)):
            f.write(f"Class {cls}: {acc:.2f}%")
            if i < len(classes) - 1:
                f.write(', ')
            else:
                f.write('\n')
        f.write(f"Mean class accuracy: {mean_class_acc:.2f}%\n")

    return {'loss': avg_loss, 'acc1': avg_acc1, 'acc5': avg_acc5}


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--epoch', type=str, default='0')

    parser.add_argument('--wandb_project', type=str, default='pruning')
    parser.add_argument('--wandb_run', type=str, default='debug')
    parser.add_argument('--wandb_entity', type=str, default='ia-gu')

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
    args.output_path = args.weight_path
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name='eval_'+args.wandb_run+'_'+args.output_path, config=vars(args))

    main(args)