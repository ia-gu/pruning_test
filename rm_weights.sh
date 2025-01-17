cd /home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/Hessian

for i in {1..500}; do
  rm */*/*/*/ckpt/$i.pth
done

cd /home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/HessianParam

for i in {1..500}; do
  rm */*/*/*/ckpt/$i.pth
done

cd /home/ueno/pruning/test/5step/debug/ResNet18/CIFAR10/L1

for i in {1..500}; do
  rm */*/*/*/ckpt/$i.pth
done
