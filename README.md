# lerobot_pusht

Train a diffusion policy in sim, then transfer to real, using the Push T task.

There are two main components of the code:

> Real (`lerobot/`) <-> sim (`diffusion_policy/`)

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

To test a model in PushT sim: `python run_sim.py --checkpoint /path/to/checkpoint`
To run the full sim2real pipeline: `python run_sim2real.py` (work in progress)
To run the open-loop version of sim2real pipeline: `python run_sim2real.py` (no feedback from PushT action)

## setup

To install the required dependencies, run:
`pip install -e .`

and `conda env update --file environment.yaml`
