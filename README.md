# pruning_test

## Overview
Pytorchのpruneライブラリをベースに枝刈りを行うテストコード

できること
- L1: L1ノルムの強度枝刈り
- Hessian: Hessian対角行列

## Use
```main.sh```で学習と枝刈り．```step```エポックごとに枝刈り+重み保存

```eval.sh```で推論．クリーンデータ，Cデータを一括で推論


