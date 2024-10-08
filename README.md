# REward BAlanced SEarch
This is an example implementation of REBASE as described in the paper [**An Empirical Analysis of Compute-Optimal
Inference for Problem-Solving with Language
Models**](https://arxiv.org/abs/2408.00724).

## Clone
    git clone --recurse-submodules git@github.com:thu-wyz/rebase.git
This command will clone our repository with the [sglang](https://github.com/sgl-project/sglang) repository as a submodule. The sglang repository should be on the *reward-model* branch, which has been modified slightly by us to support our reward model.
The benchmark datasets: [MATH](https://github.com/hendrycks/math), [GSM8K](https://github.com/openai/grade-school-math).

## Install
In order to install SGLang and other dependencies:

    cd sglang
    pip install -e "python[all]"

## Finetune
Our finetuning code for policy models and reward models is based on [gpt-accelera](https://github.com/Edward-Sun/gpt-accelera)
You can find the models on huggingface: [Llemma-7b](https://huggingface.co/tkitsers/Llemma-metamath-7b), 
[Llemma-34b](https://huggingface.co/tkitsers/Llemma-metamath-34b), [Llemma reward model](https://huggingface.co/tkitsers/Llemma-reward-model).


## Launch Server
You can use **tmux** to start the servers, or run them in the background by adding **&** at the end of the scripts.
Make sure to set the correct paths on your device.

    bash ./scripts/run_policy.sh
    bash ./scripts/run_reward.sh

## REBASE
Before starting the REBASE, set the hyperparameters in the YAML file. Then run:

    bash ./scripts/rebase.sh

## Evaluate
    bash ./scripts/evaluate.sh

## Citation
    @misc{wu2024empiricalanalysiscomputeoptimalinference,
      title={An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models}, 
      author={Yangzhen Wu and Zhiqing Sun and Shanda Li and Sean Welleck and Yiming Yang},
      year={2024},
      eprint={2408.00724},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.00724}, 
    }
