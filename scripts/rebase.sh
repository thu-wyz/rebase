set -e
set -x


POLICY=30000
REWARD=30010

export DATA_DIR=/usr1/data/yangzhew/dataset/prm800k/prm800k/math_splits/test.jsonl
export OUT_DIR=./exp_results/
export PARA_PATH=./hype-parameters/rebase.yaml



python3 rebase.py --input_path $DATA_DIR \
    --output_path $OUT_DIR/answers.json \
    --parameter_path $PARA_PATH \
    --policy_host http://localhost:$POLICY \
    --reward_host http://localhost:$REWARD 

