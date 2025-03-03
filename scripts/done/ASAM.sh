# python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/None/1.024/42/0.0/20250213111530/ckpt
# python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/None/1.024/42/0.0/20250213111530/ckpt
# python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/None/1.024/42/0.0/20250213111530/ckpt

# ASAM hyperparameter
python main.py --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=380 --step=1 --pruning_ratio=0.0 --importance=None --rho=0.5
# AdaSAP hyperparameter
# python main.py --wandb_run=ASAM --lr=1.024 --momentum=0.875 --weight_decay=3.05e-05 --optimizer=ASAM --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.0 --importance=None --rho=2.0


# python main.py --wandb_run=keepgpu --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=10000000000 --step=1 --pruning_ratio=100 --importance=None --train_method=AT