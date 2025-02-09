# pruning_test

## Overview
Pytorchのpruneライブラリをベースに枝刈りを行うテストコード

できること
- L1: L1ノルムの強度枝刈り
- Hessian: Hessian対角行列
- HesianParam: Hessian × L1ノルム
※ 完全なHessianはメモリの関係で不可能

## Evaluation
- corruption data performance ⇒ eval.sh
- loss landscape ⇒ view.sh
- fourier heat map ⇒ fourier.sh

引数
| Name | Roll|
|----|----|
| model | モデル：ResNetかVGG |
| dataset | データセット：CIFAR，Tiny，ImageNet（クラス数が合わないとエラー） |
| pruning_ratio | 枝刈りする割合：1－疎度 |
| optimizer | 最適化手法：AdamかSGD |
| train_method | 学習方法：普通か，敵対的（AT） |
| weight_path | 学習済み重みから刈る場合は入れる |
| step | 枝刈りのステップ：1なら1epochごと |
| importance | 重要度：L1，Hessian，HessianParam |



<!-- ## Note -->
<!-- 実験設定で決めないといけないこと
- 枝刈りの頻度 
    - 今は各epoch 
    - 最後に追加で20epoch学習 ⇒ サチらせるため
- 疎度
    - 今は全レイヤ固定
        - レイヤごとに変えるべきでは
    - 今は頻度に合わせて段階的に枝刈り (e.g. ```step```が10なら10回に疎度を分ける)
    - 前のマスクを保持するか，新しく刈るべきか -->
