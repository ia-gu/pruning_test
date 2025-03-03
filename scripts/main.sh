python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=L1
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=L1


# python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.7/20250228021051/ckpt
# python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.8/20250228021121/ckpt
# python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.9/20250228021152/ckpt
# python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.99/20250228021222/ckpt

# python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.7/20250228021051/ckpt
# python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.8/20250228021121/ckpt
# python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.9/20250228021152/ckpt
# python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.99/20250228021222/ckpt

# python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.7/20250228021051/ckpt
# python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.8/20250228021121/ckpt
# python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.9/20250228021152/ckpt
# python view.py --mpi --model=ResNet18 --num_classes=100 --epoch=final_weight --dataset=CIFAR100 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR100/L1/0.1/42/0.99/20250228021222/ckpt