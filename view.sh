python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=10 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/None/0.0001/42/1/20250109151522/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=10 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/HessianParam/0.0001/42/0.99/20250113165405/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=10 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/Hessian/0.0001/42/0.99/20250113124305/ckpt
