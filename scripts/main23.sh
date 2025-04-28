python main.py --seed=42 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=0.5 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Hessian
python main.py --seed=111 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=0.5 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Hessian
python main.py --seed=3407 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=0.5 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Hessian --verbose=True
python main.py --seed=42 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=0.5 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Hessian
python main.py --seed=111 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=0.5 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Hessian
python main.py --seed=3407 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=0.5 --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Hessian --verbose=True

python main.py ---seed=42 -wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=Hessian
python main.py ---seed=111 -wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=Hessian
python main.py ---seed=3407 -wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.8 --importance=Hessian --verbose=True
python main.py --seed=42 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Hessian
python main.py --seed=111 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Hessian
python main.py --seed=3407 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.9 --importance=Hessian --verbose=True
python main.py --seed=42 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Hessian
python main.py --seed=111 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Hessian
python main.py --seed=3407 --wandb_run=ASAM --lr=0.1 --optimizer=ASAM --rho=1.0 --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epochs=200 --step=1 --pruning_ratio=0.99 --importance=Hessian --verbose=True


for dir in /home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR10/Hessian/*/*/*/*/ckpt; do
    if [ -d "$dir" ]; then
        echo $dir
        python eval.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=$dir
        python fourier_heat_map.py --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --weight_path=$dir
        python view.py --mpi --model=ResNet18 --num_classes=10 --dataset=CIFAR10 --epoch=final_weight --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=$dir
    fi
done

for dir in /home/ueno/pruning/test/logs/ASAM/ResNet18/CIFAR100/Hessian/*/*/*/*/ckpt; do
    if [ -d "$dir" ]; then
        echo $dir
        python eval.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=$dir
        python fourier_heat_map.py --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --weight_path=$dir
        python view.py --mpi --model=ResNet18 --num_classes=100 --dataset=CIFAR100 --epoch=final_weight --vmin=0 --vmax=2 --x=-1:1:51 --y=-1:1:51 --dir_type=weights --xnorm=filter --xignore=biasbn --ynorm=filter --yignore=biasbn --weight_path=$dir
    fi
done