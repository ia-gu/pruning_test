# python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/Hessian/0.0001/42/0.95/20250115133943/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/Hessian/0.0001/42/0.99/20250115044918/ckpt

python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/HessianParam/0.0001/42/0.95/20250116071312/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/HessianParam/0.0001/42/0.99/20250115222511/ckpt


# python main.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=1000000000 --step=1 --pruning_ratio=100 --importance=None