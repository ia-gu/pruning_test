python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.7/20250209183123/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.8/20250209192557/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.9/20250209201948/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.99/20250209211333/ckpt

python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.7/20250209220721/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.8/20250210014135/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.9/20250210051545/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.99/20250210084928/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.7/20250209183123/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.8/20250209192557/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.9/20250209201948/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.99/20250209211333/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.7/20250209220721/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.8/20250210014135/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.9/20250210051545/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.99/20250210084928/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.7/20250209183123/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.8/20250209192557/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.9/20250209201948/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/L1/0.1/42/0.99/20250209211333/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.7/20250209220721/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.8/20250210014135/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.9/20250210051545/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/L1/0.1/42/0.99/20250210084928/ckpt

# python main.py --wandb_run=keepgpu --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=10000000000 --step=1 --pruning_ratio=100 --importance=None --train_method=AT
