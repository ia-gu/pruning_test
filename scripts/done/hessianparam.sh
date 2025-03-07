python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=HessianParam
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=HessianParam
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=HessianParam
python main.py --wandb_run=main --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=HessianParam

python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.7 --importance=HessianParam --train_method=AT
python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=HessianParam --train_method=AT
python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=HessianParam --train_method=AT
python main.py --wandb_run=AT --lr=0.1 --optimizer=SGD --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=HessianParam --train_method=AT

# python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250126122634/ckpt
# python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/from_checkpoint/ResNet18/CIFAR10/HessianParam/0.1/42/0.95/20250126103556/ckpt
# python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/from_checkpoint/ResNet18/CIFAR10/HessianParam/0.1/42/0.95/20250126103556/ckpt
# python main.py --wandb_run=keepgpu --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=10000000000 --step=1 --pruning_ratio=100 --importance=None --train_method=AT
