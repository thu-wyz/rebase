set -e
set -x
#!/bin/bash

MODEL_REPO=/usr1/data/yangzhew/models/llemma-7b_metamath_shepherd


PORT=30000
tenser_parellel_size=1

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tenser_parellel_size --trust-remote-code



