python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.7 --importance=L1
python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.8 --importance=L1
python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.9 --importance=L1
python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.99 --importance=L1
