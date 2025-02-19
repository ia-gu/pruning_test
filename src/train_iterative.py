from tqdm import tqdm
import torch
import torch.nn.utils.prune as prune
import os
import gc
import wandb
import math

from src.utils import ASAM, WarmupCosineScheduler
from src.prune_model import prune_model

def train_iterative(args, model, train_loader, eval_loader, criterion, device):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        minimizer = None
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[60, 120, 160, 190])
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        minimizer = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.optimizer == 'ASAM':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        minimizer = ASAM(optimizer, model, rho=args.rho, eta=0.01)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=8, total_epochs=args.epochs, base_lr=args.lr)
    else:
        raise ValueError('Optimizer should be either SGD, Adam or ASAM')
    
    # ----- ウォームアップフェーズ：最初の10エポックはプルーニングなし -----
    warmup_epochs = 10
    total_epochs = args.epochs  # 例：90エポック（=10エポック＋80エポック）
    for epoch in range(warmup_epochs):
        train_loss, train_accuracy = train_epoch(epoch, model, train_loader, criterion, device, optimizer, minimizer)
        eval_loss, eval_accuracy = eval_epoch(epoch, model, criterion, eval_loader, device)
        with open(os.path.join(args.output_path, 'result.txt'), 'a') as f:
            f.write(f'Epoch: |{epoch+1}| Train_Loss: |{train_loss}| Train_Accuracy: |{train_accuracy}| '
                    f'Eval_Loss: |{eval_loss}| Eval_Accuracy: |{eval_accuracy}|\n')
        wandb.log({'epoch': epoch+1, 'Train_Loss': train_loss, 'Train_Accuracy': train_accuracy,
                   'Eval_Loss': eval_loss, 'Eval_Accuracy': eval_accuracy})
        scheduler.step()
    
    # dense なモデルの総ニューロン数 m を計算（全 weight の要素数）
    m = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            m += module.weight.numel()
    
    # ----- プルーニングフェーズ：ウォームアップ後、80エポックでプルーニングを実施 -----
    fine_tune_epochs = total_epochs - warmup_epochs  # 例：80エポック
    total_finetune_iterations = fine_tune_epochs * len(train_loader)
    R = total_finetune_iterations // 30  # 総プルーニング回数（30イテレーションごと）
    prune_iter_count = 0
    global_finetune_iter = 0
    for epoch in range(warmup_epochs, total_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for (inputs, targets) in progress_bar:
            global_finetune_iter += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if minimizer:
                minimizer.ascent_step()
                criterion(model(inputs), targets).mean().backward()
                minimizer.descent_step()
            for module in model.modules():
                if hasattr(module, 'weight_mask'):
                    with torch.no_grad():
                        if module.weight.grad is not None:
                            module.weight.grad *= module.weight_mask
            optimizer.step()
            for module in model.modules():
                if hasattr(module, 'weight_mask'):
                    with torch.no_grad():
                        module.weight *= module.weight_mask

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_description(
                f"Epoch {epoch+1} | Iter {global_finetune_iter} | Loss: {round(loss.item(),3)} | Acc: {round(100.*correct/total,3)}"
            )

            # 30イテレーションごとにプルーニングを実施
            if global_finetune_iter % 30 == 0:
                prune_iter_count += 1
                alpha = prune_iter_count / R if R > 0 else 1.0
                # 最終的に p 削除するので、残存率は k_final = 1 - p
                k_final = 1 - args.pruning_ratio  
                # r 回目のプルーニング後、目標とするニューロン数は： m * (k_final)^alpha
                target_neurons = m * (k_final ** alpha)
                # 現在の残存ニューロン数（各 weight の非ゼロ要素数の和）
                current_neurons = 0
                for module in model.modules():
                    if hasattr(module, 'weight'):
                        current_neurons += (module.weight != 0).sum().item()
                # 今回削る割合： (current - target) / current
                pr_fraction = (current_neurons - target_neurons) / current_neurons
                pr_fraction = max(0.0, min(pr_fraction, 1.0))
                print(f"Pruning at iter {global_finetune_iter}: r = {prune_iter_count}/{R}, "
                      f"alpha = {alpha:.4f}, current = {current_neurons}, "
                      f"target = {target_neurons:.2f}, pr_fraction = {pr_fraction:.4f}")
                model, tmp_model = pruning(args, epoch, model, train_loader, eval_loader, criterion, pr_fraction, device)
                torch.save(tmp_model.state_dict(), os.path.join(args.output_path, 'ckpt', 
                                f'checkpoint_iter_epoch{epoch+1}_iter{global_finetune_iter}.pth'))
                del tmp_model
                gc.collect()
        
        avg_train_loss = round(epoch_loss / len(train_loader), 3)
        train_accuracy = round(100. * correct / total, 3)
        eval_loss, eval_accuracy = eval_epoch(epoch, model, criterion, eval_loader, device)
        with open(os.path.join(args.output_path, 'result.txt'), 'a') as f:
            f.write(f'Epoch: |{epoch+1}| Train_Loss: |{avg_train_loss}| Train_Accuracy: |{train_accuracy}| '
                    f'Eval_Loss: |{eval_loss}| Eval_Accuracy: |{eval_accuracy}|\n')
        wandb.log({'epoch': epoch+1, 'Train_Loss': avg_train_loss, 'Train_Accuracy': train_accuracy,
                   'Eval_Loss': eval_loss, 'Eval_Accuracy': eval_accuracy})
        scheduler.step()
    
    # ----- 最終的に各モジュールからプルーニングマスクを除去し，最終モデルを保存 -----
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
    torch.save(model.state_dict(), os.path.join(args.output_path, 'ckpt', 'final_weight.pth'))

def train_epoch(epoch, model, loader, criterion, device, optimizer, minimizer=None):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    for (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if minimizer:
            minimizer.ascent_step()
            criterion(model(inputs), targets).mean().backward()
            minimizer.descent_step()
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                with torch.no_grad():
                    if module.weight.grad is not None:
                        module.weight.grad *= module.weight_mask
        optimizer.step()
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                with torch.no_grad():
                    module.weight *= module.weight_mask

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_description(
            f"Epoch {epoch+1} | Loss: {round(loss.item(),3)} | Acc: {round(100.*correct/total,3)}"
        )
    return round(train_loss/len(loader), 3), round(100.*correct/total, 3)

def eval_epoch(epoch, model, criterion, loader, device):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f"Eval Epoch {epoch+1}")
    with torch.no_grad():
        for (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_description(
                f"Epoch {epoch+1} | Loss: {round(loss.item(),3)} | Acc: {round(100.*correct/total,3)}"
            )
    return round(eval_loss/len(loader), 3), round(100.*correct/total, 3)

def pruning(args, epoch, model, train_loader, eval_loader, criterion, pr_fraction, device):
    print('--- プルーニング前のモデル ---')
    total_params = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            non_zero_params = (module.weight != 0).sum().item()
            total_params += non_zero_params
    print(f'Total non-zero parameters before pruning: {total_params:,}')

    model, tmp_model = prune_model(args, model, criterion, train_loader, amount=pr_fraction)
    
    total_params_after = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            non_zero_params = (module.weight != 0).sum().item()
            total_params_after += non_zero_params
    print(f'Total non-zero parameters after pruning: {total_params_after:,}')

    eval_loss, eval_accuracy = eval_epoch(epoch, model, criterion, eval_loader, device)
    print(f'Epoch: {epoch+1} | Pruning Eval_Loss: {eval_loss} | Pruning Eval_Accuracy: {eval_accuracy}')
    wandb.log({'epoch': epoch+1, 'Pruning_Loss': eval_loss, 'Pruning_Accuracy': eval_accuracy,
               'Total_parameters': total_params_after})
    return model, tmp_model
