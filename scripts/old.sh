# Adam，敵対的学習
python main.py --optimizer=Adam --wandb_run=5step --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=1000 --step=5 --pruning_ratio=0.99 --importance=Hessian --train_method=AT

# 刈らない
python main.py --wandb_run=tuning --lr=0.1 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.0 --importance=None

# 学習済み重みから枝刈り
python main.py --lr=0.1 --optimizer=SGD --wandb_run=from_ckpt --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=HessianParam --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/None/0.1/42/0.0/20250119062520/ckpt/final_weight.pth

# テスト
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/5step/ResNet18/CIFAR10/Hessian/0.0001/42/0.99/20250122055530/ckpt

# keep用
python main.py --wandb_run=keepgpu --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=10000000000 --step=1 --pruning_ratio=100 --importance=None --train_method=AT

# フーリエヒートマップ
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.9999/20250122111143/ckpt

# 損失地形
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.9999/20250122111143/ckpt
