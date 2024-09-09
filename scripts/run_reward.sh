set -e
set -x
#!/bin/bash


MODEL_REPO="reward-repo"

PORT=30010
tenser_parellel_size=1

CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tenser_parellel_size --trust-remote-code



