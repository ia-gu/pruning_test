# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import random
import numpy as np

import torch

from vit_wdpruning import VisionTransformerWithWDPruning
from tqdm import tqdm

from datasets import build_robust_test_dataset
from engine import evaluate_robustness

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def count_parameters(model):
    params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'blocks' in n)
    return params

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def valid(args, model,test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          )
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits.view(-1, 10), y.view(-1))
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--arch', default='deit_small', type=str)
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz", help="Where to search for pretrained ViT models.")
    parser.add_argument('--data-path', default='./data', type=str)
    parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'TINY', 'INAT', 'INAT19'], type=str, help='Image Net dataset path')
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--distill', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument("--classifiers", type=int, nargs='+', default=[8,10])
    parser.add_argument("--classifier_choose", default=12, type=int)
    args = parser.parse_args()
    set_seed(args)
    device = torch.device("cuda")
    args.device = device
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    with open(args.pretrained_dir.replace('checkpoint.pth', 'test_log.txt'), "a") as f:
        f.write(str(args) + "\n")

    if args.data_set == 'CIFAR10': args.nb_classes = 10
    elif args.data_set == 'CIFAR100': args.nb_classes = 100
    elif args.data_set == 'TINY': args.nb_classes = 200
    else: args.nb_classes = 1000

    if args.arch == 'deit_small':
        embed_dim = 384
        num_heads = 6
    elif args.arch == 'deit_tiny':
        embed_dim = 192
        num_heads = 3

    model = VisionTransformerWithWDPruning(num_classes=args.nb_classes,
                                         patch_size=16, embed_dim=embed_dim, depth=12, num_heads=num_heads, mlp_ratio=4,
                                         qkv_bias=True, distilled=args.distill, classifiers=args.classifiers,
                                         classifier_choose=args.classifier_choose
                                         )

    # size of unpruned model
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter of original model: \t%2.1fM" % (num_params/1000000))

    # model.load_state_dict(torch.load(args.pretrained_dir)['model'])

    model.LayerPruningAndLoadParams(dir=args.pretrained_dir)

    model.eval()

    model._make_structural_pruning()

    total_num_params = 0
    for name, param in model.named_parameters():
        if 'blocks' in name:
            total_num_params += (param.abs() > 1e-8).sum()

    model.to(args.device)

    # warmup!!!
    # test_stats = evaluate_classifiers(data_loader_val, model, device,classifiers = args.classifiers)
    # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    # do real measurement

    print('*' * 100)
    print('Num of Parameters: ', total_num_params)
    print(
        f'Remaining Parameters as compared to baseline: {(total_num_params/num_params*100):.2f}%')
    print('*' * 100)
    with open(args.pretrained_dir.replace('checkpoint.pth', 'test_log.txt'), "a") as f:
        f.write("Total Parameter of original model: \t%2.1fM" % (num_params/1000000))
        f.write('*' * 100 + '\n')
        f.write('Num of Parameters: ' + str(total_num_params) + '\n')
        f.write(
            f'Remaining Parameters as compared to baseline: {(total_num_params/num_params*100):.2f}%\n')
        f.write('*' * 100 + '\n')


    if args.eval:
        datasets = build_robust_test_dataset(args=args)
        dataset_types = ['clean', 'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 
                         'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
        for i in range(len(datasets)):
            if not len(datasets) == len(dataset_types):
                raise ValueError('Datasets and dataset_types should have the same length')
            if datasets[i] == None:
                continue
            data_loader_val = torch.utils.data.DataLoader(
                datasets[i],
                batch_size=int(args.eval_batch_size),
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )
            evaluate_robustness(data_loader_val, model, device, args, dataset_types[i])

if __name__ == "__main__":
    main()
