python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250220141600/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250220163906/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250220190046/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250220212320/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250220141600/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250220163906/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250220190046/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250220212320/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250220141600/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250220163906/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250220190046/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250220212320/ckpt


python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.7 --importance=Hessian
python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.8 --importance=Hessian
python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.9 --importance=Hessian
python main.py --wandb_run=ASAM400 --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.99 --importance=Hessian