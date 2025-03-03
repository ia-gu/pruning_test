python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/None/0.1/42/0.0/20250228021055/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/None/0.1/42/0.0/20250228021055/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/None/0.1/42/0.0/20250228021055/ckpt


python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.7/20250228025419/ckpt
python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.8/20250228043304/ckpt
python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.9/20250228061046/ckpt
python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.99/20250228074812/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.7/20250228025419/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.8/20250228043304/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.9/20250228061046/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.99/20250228074812/ckpt

python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.7/20250228025419/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.8/20250228043304/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.9/20250228061046/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/Hessian/0.1/42/0.99/20250228074812/ckpt


python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.7/20250228021053/ckpt
python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.8/20250228035108/ckpt
python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.9/20250228053131/ckpt
python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.99/20250228071117/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.7/20250228021053/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.8/20250228035108/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.9/20250228053131/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.99/20250228071117/ckpt

python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.7/20250228021053/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.8/20250228035108/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.9/20250228053131/ckpt
python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/HessianParam/0.1/42/0.99/20250228071117/ckpt
