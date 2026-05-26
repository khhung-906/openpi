"""
FOR FIXED STATE, REMEMBER TO CHANGE THE STD OF STATE AND ACTION IN THE CONFIG!!!!!!!!!!!!

q0, q99!!!!!!!!!!!!
CHANGE STD TO 1!!!!!!!!!!!!!!!!
"""

source .venv/bin/activate

repo_id="johnson906/droid_flower_insert_50"
uv run scripts/compute_norm_stats.py \
    --config-name expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --repo-id $repo_id
