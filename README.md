# lerobot_pusht

Train a diffusion policy in sim, then transfer to real, using the Push T task.

There are two main components of the code:

> Real (`lerobot/`) <-> sim (`diffusion_policy/`)

For the lerobot, the scripts for calibration, control, data collection are in `lerobot/scripts/pusht`. Data is collected and uploaded to HuggingFace. Then train an imitation learning policy in real using `lerobot/scripts/train.py`:

```
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_test \
  --policy.type=diffusion \
  --output_dir=outputs/train/act_koch_test \
  --job_name=act_koch_test \
  --device=cuda \
  --wandb.enable=true
```

For the sim component, we add an additional vanilla PushT task (`PushTEnv`) to the diffusion policy repo. The vanilla Push T takes in 5 observations `(x,y,x_T,y_T,theta_T)` instead of 9 for the keypoint. We train a diffusion policy in sim:

```
python train.py --config-dir=. \
  --config-name=train_diffusion_transformer_lowdim_pusht_workspace.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}*${name}_${task_name}'
```

Finally, we experiment with sim2real with transferring the learned model into real Lerobot environment.
