python src/main.py \
    --seed=0 \
    --mode="inference" \
    --data_dir="data" \
    --model_type="gpt2" \
    --pad_token="<pad>" \
    --bos_token="<bos>" \
    --eos_token="<eos>" \
    --sp1_token="<sp1>" \
    --sp2_token="<sp2>" \
    --gpu="0" \
    --max_len=1024 \
    --max_turns=5 \
    --top_p=0.9 \
    --ckpt_dir="saved_models" \
    --ckpt_name=CKPT_NAME \
    --end_command="Abort!"