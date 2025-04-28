python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=L1

python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=L1 --train_method=AT
python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=L1 --train_method=AT
python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=L1 --train_method=AT
python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=L1 --train_method=AT
