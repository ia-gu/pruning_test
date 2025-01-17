# python main.py --output_path=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=500 --step=5 --pruning_ratio=0.99 --importance=Hessian
# python main.py --output_path=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=500 --step=5 --pruning_ratio=0.95 --importance=Hessian
# python main.py --output_path=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=500 --step=5 --pruning_ratio=0.99 --importance=HessianParam
# python main.py --output_path=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=500 --step=5 --pruning_ratio=0.95 --importance=HessianParam
# python main.py --output_path=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=500 --step=5 --pruning_ratio=0.99 --importance=L1
# python main.py --output_path=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=500 --step=5 --pruning_ratio=0.95 --importance=L1

python main.py --wandb_run=keep --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=1000000000 --step=1 --pruning_ratio=100 --importance=None
