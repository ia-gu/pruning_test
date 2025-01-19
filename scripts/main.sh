# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.0 --importance=None

# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.9 --importance=Hessian
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.9 --importance=HessianParam
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.9 --importance=L1
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.99 --importance=Hessian
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.95 --importance=Hessian
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.99 --importance=HessianParam
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.95 --importance=HessianParam
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.99 --importance=L1
# python main.py --wandb_run=32*32 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.95 --importance=L1

# python main.py --wandb_run=AdversarialTraining --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=1000 --step=1 --pruning_ratio=100 --importance=None --train_method=AT
# python main.py --wandb_run=AdversarialTraining --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=100 --importance=None --train_method=AT
# python main.py --wandb_run=AdversarialTraining --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=300 --step=1 --pruning_ratio=100 --importance=None --train_method=AT

python main.py --wandb_run=tuning --lr=0.1 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=100 --importance=None
# python main.py --wandb_run=tuning --lr=0.01 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=100 --importance=None

# python main.py --wandb_run=keepgpu --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=10000000000 --step=1 --pruning_ratio=100 --importance=None