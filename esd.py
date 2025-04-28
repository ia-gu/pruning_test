import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import sys
sys.path.append('./')
import argparse
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.get_dataset import build_dataset
from src.model import CNNModel

def hvp_full_dataset(model, loss_fn, data_loader, v):
    """
    全データ上の Hessian-vector product をバッチ単位で累積計算し、VRAM OOM を回避。
    """
    device = v.device
    total_samples = len(data_loader.dataset)
    hvp = torch.zeros_like(v, device=device)

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 順伝播と損失
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        # 一階勾配
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        # Hv = ∇(grad_vec · v)
        grad_dot_v = torch.dot(grad_vec, v)
        hvp_batch = torch.autograd.grad(grad_dot_v, model.parameters(), retain_graph=False)
        hvp_batch_vec = torch.cat([h.contiguous().view(-1) for h in hvp_batch])
        # バッチ重みづけで累積
        hvp += (inputs.size(0) / total_samples) * hvp_batch_vec.detach()
        torch.cuda.empty_cache()

    return hvp

def compute_hessian_spectrum(model, train_loader, num_eigenvalues=50):
    """
    Lanczos アルゴリズムによる Hessian 固有値スペクトル近似。
    全データ損失に基づく Hessian-vector product を用い、
    フル再直交化を行うことで安定化。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    device = next(model.parameters()).device

    # 損失関数と mv 定義
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    mv = lambda q: hvp_full_dataset(model, loss_fn, train_loader, q)

    # Lanczos 反復回数
    k = min(num_eigenvalues, num_params)

    # 初期ベクトルと係数
    q = torch.randn(num_params, device=device)
    q = q / torch.norm(q)
    q_list = []
    alpha = torch.zeros(k, device=device)
    beta  = torch.zeros(k, device=device)

    # Lanczos ループ
    for i in tqdm(range(k), desc='Lanczos iterations'):
        v = mv(q)
        alpha[i] = torch.dot(q, v)

        # 基本的な差し引き
        v = v - alpha[i] * q
        if i > 0:
            v = v - beta[i-1] * q_prev

        # フル再直交化
        for qj in q_list:
            v = v - torch.dot(qj, v) * qj

        beta[i] = torch.norm(v)
        q_prev = q
        q = v / (beta[i] + 1e-12)
        q_list.append(q_prev)

    # 三重対角行列の固有値を計算
    T = torch.diag(alpha) + torch.diag(beta[:-1], 1) + torch.diag(beta[:-1], -1)
    eigenvalues, _ = torch.linalg.eigh(T)
    return eigenvalues

def main(args):
    set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルのロード
    model = CNNModel(model=args.model, classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.weight_path, f"{args.epoch}.pth")))
    model = model.to(device)

    # 非ゼロパラメータ数
    nonzero_params = sum((param != 0).sum().item() for param in model.parameters())
    print(f"Non-zero parameters: {nonzero_params}")

    # データセット／ローダー
    train_dataset, _ = build_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2
    )

    # Hessian スペクトル計算
    num_eigs = 100
    eigenvalues = compute_hessian_spectrum(model, train_loader, num_eigenvalues=num_eigs)
    eig = eigenvalues.cpu().numpy()

    os.makedirs(args.weight_path.replace(args.weight_path.split('/')[-1], os.path.join('hessian_esd', args.epoch)), exist_ok=True)

    # 1) 分布のヒストグラムをプロット
    plt.figure()
    plt.hist(eig, bins=50)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Hessian Spectrum Distribution')
    plt.tight_layout()
    plt.savefig(args.weight_path.replace(args.weight_path.split('/')[-1], os.path.join('hessian_esd', args.epoch, 'hessian_esd.png')), dpi=100)
    plt.savefig(args.weight_path.replace(args.weight_path.split('/')[-1], os.path.join('hessian_esd', args.epoch, 'hessian_esd.svg')), dpi=100)

    # 2) bulk ratio（シャープネス）
    sharpness = eig[-1] / eig[-5]
    with open(args.weight_path.replace(args.weight_path.split('/')[-1], os.path.join('hessian_esd', args.epoch, 'hessian_esd.txt')), 'a') as f:
        f.write(f"λ_max: {eig[-1]:.8f}, λ_5th: {eig[-5]:.8f}, Sharpness (λ_max / λ_5th): {sharpness:.8f}\n")
        f.write(f"Eigenvalues: {eig}\n")
    print(f"Eigenvalues: {eig}")
    print(f"λ_max: {eig[-1]:.8f}, λ_5th: {eig[-5]:.8f}")
    print(f"Sharpness (λ_max / λ_5th): {sharpness:.8f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--epoch', type=str, default='0')
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
    output_csv = args.weight_path.replace(
        os.path.basename(args.weight_path),
        os.path.join('hessian_esd', args.epoch, 'hessian_esd.txt')
    )
    if os.path.exists(output_csv):
        print('Already Calculated')
    else:
        main(args)

