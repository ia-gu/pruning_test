from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import os
import wandb

from src.utils.prune_model import prune_model

def train(args, model, train_loader, eval_loader, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch/args.warmup_epochs if epoch < args.warmup_epochs else 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))))
    num_steps = args.epochs // args.step
    pruning_ratio = 0.0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_epoch(epoch, model, train_loader, criterion, device, optimizer, scheduler)
        eval_loss, eval_accuracy = eval_epoch(epoch, model, criterion, eval_loader, device)
        with open(os.path.join(args.output_path, 'result.txt'), 'a') as f:
            f.write(f'Epoch: |{epoch+1}| Train_Loss: |{train_loss}| Train_Accuracy: |{train_accuracy}| Eval_Loss: |{eval_loss}| Eval_Accuracy: |{eval_accuracy}|\n')
        wandb.log({'epoch': epoch+1, 'Train_Loss': train_loss, 'Train_Accuracy': train_accuracy, 'Eval_Loss': eval_loss, 'Eval_Accuracy': eval_accuracy})

        if (epoch+1) % args.step == 0:
            pruning_ratio += (args.pruning_ratio / num_steps)
            model = pruning(epoch, model, eval_loader, criterion, pruning_ratio, device)
            # save model
            torch.save(model.state_dict(), os.path.join(args.output_path, str(epoch+1)+'.pth'))


def train_epoch(epoch, model, loader, criterion, device, optimizer, scheduler):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(loader)

    for (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for _, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                with torch.no_grad():
                    if module.weight.grad is not None:
                        module.weight.grad *= module.weight_mask
        optimizer.step()
        for _, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                with torch.no_grad():
                    module.weight *= module.weight_mask
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_description(f'Epoch: {epoch+1} | Train_Loss: {round(train_loss/len(loader), 3)} | Train_Accuracy: {round(100.*correct/total, 3)}')

    return round(train_loss/len(loader), 3), round(correct/total, 3)

def eval_epoch(epoch, model, criterion, loader, device):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(loader)

    with torch.no_grad():
        for (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_description(f'Epoch: {epoch+1} | Eval_Loss: {round(eval_loss/len(loader), 3)} | Eval_Accuracy: {round(100.*correct/total, 3)}')
    
    return round(eval_loss/len(loader), 3), round(correct/total, 3)

def pruning(epoch, model, eval_loader, criterion, pruning_ratio, device):
    print('Pruning前のモデル:')
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f'\nTotal number of non-zero Conv2d parameters: {total_params:,}')

    model = prune_model(model, amount=pruning_ratio)
    print('Pruning後のモデル:')
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f'\nTotal number of non-zero Conv2d parameters: {total_params:,}')

    eval_loss, eval_accuracy = eval_epoch(epoch, model, criterion, eval_loader, device)
    print(f'Epoch: {epoch+1} | Pruning_Loss: {eval_loss} | Pruning_Accuracy: {eval_accuracy}')
    wandb.log({'epoch': epoch+1, 'Pruning_Loss': eval_loss, 'Pruning_Accuracy': eval_accuracy, 'Total parameters': total_params})
    return model