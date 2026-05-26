source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_id="droid_pick_cube_10"
repo_id="johnson906/$data_id"
assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state"
assets_id="johnson906/droid_pick_cube_15"

# Proxy norm stats: closest available is light2_30; run compute_norm_stats for light2_25 if distributions differ.
uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name=${data_id}_lora_sft \
    --overwrite \
    --data.repo_id=$repo_id \
    --data.assets.assets_dir=$assets_dir \
    --data.assets.asset_id=$assets_id \
    --num_train_steps=4001 \
    --save_interval=2000 \
    --fsdp_devices=1