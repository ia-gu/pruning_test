python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=L1

python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=Jacobian
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=Jacobian
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Jacobian
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Jacobian
