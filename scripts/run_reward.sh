set -e
set -x
#!/bin/bash


MODEL_REPO=/usr1/data/yangzhew/models/shepherd-34b-finetune/shepherd-34b-hf-6000

PORT=30010
tenser_parellel_size=2

CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tenser_parellel_size --trust-remote-code



