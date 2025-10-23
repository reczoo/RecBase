
python amazon_text_emb.py \
    --dataset Games \
    --root path_to_features \
    --gpu_id 1 \
    --plm_name llama2 \
    --plm_checkpoint "Llama-2-7b-chat-hf" \
    --max_sent_len 2048 \
    --word_drop_ratio -1