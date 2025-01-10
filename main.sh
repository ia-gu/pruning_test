# python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.5
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.8 --importance=Hessian
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.7 --importance=Hessian
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.6 --importance=Hessian
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.9 --importance=L1
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.8 --importance=L1
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.7 --importance=L1
python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=100 --step=1 --pruning_ratio=0.6 --importance=L1

python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/Hessian/0.0001/42/0.9/20250109185134/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=100 --weight_path=/home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/Hessian/0.0001/42/0.9/20250109185134/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/None/0.0001/42/0.5/20250109151522/ckpt

python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=1000000000 --step=1 --pruning_ratio=100 --importance=None
