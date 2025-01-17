cd /home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/Hessian

for i in {99..100}; do
  rm */*/*/*/ckpt/$i.pth
done

cd /home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/HessianParam

for i in {99..100}; do
  rm */*/*/*/ckpt/$i.pth
done

cd /home/ueno/pruning/test/logs/debug/ResNet18/CIFAR10/L1

for i in {99..100}; do
  rm */*/*/*/ckpt/$i.pth
done
