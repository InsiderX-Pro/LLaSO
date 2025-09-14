#!/bin/bash

################################################################################
# LLaSO Stage 2: Multi-task Instruction Tuning
#
# This script fine-tunes the LLaSO model on multi-task instruction data
# It uses the Stage 1 aligned audio–text projector as initialization and optimizes both the LLM and projector.
#
# Key arguments:
# --model_name_or_path       Path to the base LLM backbone (Llama-3.2-3B-Instruct).
# --audio_tower              Path to the audio encoder (Whisper-large-v3).
# --data_path                Stage 2 training dataset (JSON with multiple tasks).
# --pretrain_audio_aligner   Path to the pretrained audio aligner from Stage 1 (e.g., /llaso/llaso_aligned_3b_ckpts/checkpoint-xxx/mm_audio_aligner.bin).
# --mm_use_audio_start_end   Insert [AUDIO_START]/[AUDIO_END] tokens (kept True).
#
# Notes:
# - Replace all paths (/code/syr/... and /llaso/...) with your own local paths.
# - Output will be saved to --output_dir (default: /llaso/llaso_sft_3b_ckpts).
# - Gradient checkpointing is enabled here (saves memory, slightly slower).
# - Learning rate is much smaller than Stage 1 (3e-5 vs. 1e-3).
################################################################################

deepspeed /llaso/train/train_mem.py \
    --deepspeed /llaso/scripts/zero2.json \
    --model_name_or_path /code/syr/Llama-3.2-3B-Instruct\
    --version llama32_audio_v1_mmtag \
    --data_path  ./LLaSO-Instruct/merged/stage2_train.json \
    --audio_tower /code/syr/whisper-large-v3 \
    --pretrain_audio_aligner /llaso/llaso_aligned_3b_ckpts/checkpoint-xxx/mm_audio_aligner.bin \
    --audio_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter  False \
    --tune_mm_audio_projector False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_audio_start_end True \
    --bf16 True \
    --output_dir /llaso/llaso_sft_3b_ckpts \
    --num_train_epochs  1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 50 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False\
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --mix_va False \
    --group_by_modality_length True \
    --max_grad_norm 1.0 \