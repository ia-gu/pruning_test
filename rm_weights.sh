cd /home/ueno/pruning/test/logs/32*32/ResNet18/CIFAR10/Hessian

for i in {1..100}; do
  rm */*/*/*/ckpt/$i.pth
done

cd /home/ueno/pruning/test/logs/32*32/ResNet18/CIFAR10/HessianParam

for i in {1..100}; do
  rm */*/*/*/ckpt/$i.pth
done

cd /home/ueno/pruning/test/logs/32*32/ResNet18/CIFAR10/L1

for i in {1..100}; do
  rm */*/*/*/ckpt/$i.pth
done
