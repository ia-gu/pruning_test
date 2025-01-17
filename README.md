# pruning_test

## Overview
Pytorchのpruneライブラリをベースに枝刈りを行うテストコード

できること
- L1: L1ノルムの強度枝刈り
- Hessian: Hessian対角行列
- HesianParam: Hessian × L1ノルム

## Evaluation
- corruption data performance ⇒ eval.sh
- loss landscape ⇒ view.sh
- fourier heat map ⇒ fourier.sh



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
