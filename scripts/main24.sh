python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250209183127/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250209200944/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250209214739/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250209232459/ckpt

python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250210010223/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250210051715/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250210093132/ckpt
python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250210134554/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250209183127/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250209200944/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250209214739/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250209232459/ckpt

python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250210010223/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250210051715/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250210093132/ckpt
python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250210134554/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250209183127/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250209200944/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250209214739/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/main/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250209232459/ckpt

python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.7/20250210010223/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.8/20250210051715/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.9/20250210093132/ckpt
python view.py --mpi --model=ResNet18 --num_classes=10 --epoch=final_weight --dataset=CIFAR10 --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=/home/ueno/pruning/test/logs/AT/ResNet18/CIFAR10/HessianParam/0.1/42/0.99/20250210134554/ckpt

# python main.py --wandb_run=keepgpu --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=10000000000 --step=1 --pruning_ratio=100 --importance=None --train_method=AT
