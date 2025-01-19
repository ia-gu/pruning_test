import torch


import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.model import CNNModel
from src.get_dataset import build_eval_dataset



def plot_fig(args, scores):

    # plt.rcParams['font.family'] = 'Times' # font familyの設定
    plt.rcParams['font.size'] = 20
    fig, ax=plt.subplots()
    ax =sns.heatmap(
        scores,
        vmin=0,
        vmax=1.0,
        cmap="jet",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    #sns.set(font_scale = 2)
    ax.collections[0].colorbar.set_label('Error')
    plt.savefig(args.weight_path.replace(args.weight_path.split('/')[-1], 'fourier_heat_map.png'), dpi=100)
    plt.savefig(args.weight_path.replace(args.weight_path.split('/')[-1], 'fourier_heat_map.svg'), dpi=100)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parameter
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--eps', type=float, default=4)
    args = parser.parse_args()

    device = torch.device('cuda')

    if not os.path.exists(args.weight_path.replace(args.weight_path.split('/')[-1], 'fourier_heat_map.csv')):

        model = CNNModel(model=args.model, classes=args.num_classes, pretrained=False)
        model.load_state_dict(torch.load(os.path.join(args.weight_path, args.epoch+'.pth')))
        model.to(device)

        csv = np.zeros([32,32])
        best_score = 0
        for xx in range(32):
            for yy in range(32):
                # x = xx*7 + 4; y = yy*7 + 4
                test_dataset, _ = build_eval_dataset(args, xx, yy, fourier=True)
                # test_dataset, _ = build_eval_dataset(args, x, y, fourier=True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250, shuffle=False, num_workers=2, pin_memory=True)
                valid_loss = 0.
                valid_acc = 0.
                valid_total = 0.

                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        inputs, targets = (b.to(device) for b in batch)

                        predictions = model(inputs)
                        correct = torch.argmax(predictions, 1) == targets
                        valid_acc += correct.sum().item()
                        valid_total += targets.size(0)
                error =  1 - valid_acc/valid_total
                csv[xx,yy] = error


                np.savetxt(args.weight_path.replace(args.weight_path.split('/')[-1], 'fourier_heat_map.csv'), csv, delimiter=',')
                print(f'H:{xx+1:d} | W:{yy+1:d} | error: {error:.3f}')

    scores = np.loadtxt(args.weight_path.replace(args.weight_path.split('/')[-1], 'fourier_heat_map.csv'), delimiter=',')
    plot_fig(args, scores)

    
