python main.py --wandb_run=main --optimizer=SGD --lr=0.1 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=Jacobian --verbose=True
python main.py --wandb_run=main --optimizer=SGD --lr=0.1 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=JacobianParam --verbose=True
