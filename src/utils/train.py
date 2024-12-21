from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch

def train(args, model, train_loader, eval_loader, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch/args.warmup_epochs if epoch < args.warmup_epochs else 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))))

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_epoch(epoch, model, train_loader, criterion, device, optimizer, scheduler)
        eval_loss, eval_accuracy = eval_epoch(epoch, model, criterion, eval_loader, device)
        print(f'Epoch: {epoch} | Train_Loss: {train_loss} | Train_Accuracy: {train_accuracy} | Eval_Loss: {eval_loss} | Eval_Accuracy: {eval_accuracy}')

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
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_description(f'Epoch: {epoch} | Train_Loss: {round(train_loss/len(loader), 3)} | Train_Accuracy: {round(100.*correct/total, 3)}')
    
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
            progress_bar.set_description(f'Epoch: {epoch} | Eval_Loss: {round(eval_loss/len(loader), 3)} | Eval_Accuracy: {round(100.*correct/total, 3)}')
    
    return round(eval_loss/len(loader), 3), round(correct/total, 3)