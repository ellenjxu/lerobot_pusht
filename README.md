# lerobot_pusht

Train a diffusion policy in sim, then transfer to real, to solve the Push T task!

![digitaltwin](https://github.com/user-attachments/assets/2d6d1b8f-ed30-4238-bbf7-f5f62a5350d9)

## setup

Hardware setup using the [Koch v1.1 arm](https://github.com/jess-moss/koch-v1-1?tab=readme-ov-file).

<img src="https://github.com/user-attachments/assets/4b5d3651-4f4e-483f-a20d-24ec2b7c36ad" width="400" alt="lerobot_setup">

To install the required dependencies, run:
`pip install -e .`
and `conda env update --file environment.yaml`

## usage

There are two main components of the code:

> Real (`lerobot/`) <-> sim (`diffusion_policy/`)
>
> ![image](https://github.com/user-attachments/assets/9e41a651-6636-4b41-ae7f-a4afec941a0b)


For the lerobot, the scripts for calibration, control, data collection are in `lerobot/scripts/`. Data is collected and uploaded to HuggingFace. Then train an imitation learning policy in real using `lerobot/scripts/train.py`:

```
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_test \
  --policy.type=diffusion \
  --output_dir=outputs/train/act_koch_test \
  --job_name=act_koch_test \
  --device=cuda \
  --wandb.enable=true
```

For the sim component, we add an additional vanilla PushT task (`PushTEnv`) to the diffusion policy repo. The vanilla Push T takes in 5 observations `(x,y,x_T,y_T,theta_T)` instead of 9 for the keypoint. To train a diffusion policy in sim:

```
python train.py --config-dir=. \
  --config-name=train_diffusion_transformer_lowdim_pusht_workspace.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}*${name}_${task_name}'
```

Finally, we experiment with sim2real transfer from the learned simulation model into real Lerobot environment. The "real2sim2real" approach maps from the real world to digital twin in sim.

1. To test a model in PushT sim: `python run_pusht.py --checkpoint /path/to/checkpoint`

2. To run the full sim2real pipeline: `python run_pusht2real.py` (work in progress)

3. To run the open-loop version of sim2real pipeline: `python run_pusht2real.py` (no feedback from PushT action

## results

rollout of diffusion policy sim2real:

https://github.com/user-attachments/assets/644d7880-7b31-49aa-93ba-59f170319a88

https://github.com/user-attachments/assets/97dde5f1-1ea9-409b-90d2-aface850fd32

## acknowledgements

- Hardware from [Koch v1.1](https://github.com/jess-moss/koch-v1-1?tab=readme-ov-file)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/tree/main)
- [Lerobot](https://github.com/huggingface/lerobot)
