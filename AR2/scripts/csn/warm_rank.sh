echo "start ce model warmup"

cd ../..

EXP_NAME=run_ce_model
OUT_DIR=./output/$EXP_NAME
DE_EXP_NAME=run_de_ict_triviaqa
DE_OUT_DIR=./output/$DE_EXP_NAME
DE_CKPT_NUM=40000
TB_DIR=./tensorboard_log/$EXP_NAME    # tensorboard log path

# CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9538 \
    python ./codesearch/run_ce_model.py \
    --model_type="/srv/nfs/VESO/models/ernie-2.0-base-en" \
    --max_seq_length=256 \
    --per_gpu_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --number_neg=15 \
    --learning_rate=1e-5 \
    --output_dir=$OUT_DIR \
    --origin_data_dir=$DE_OUT_DIR/$DE_CKPT_NUM/train_ce_0_triviaqa.json \
    --origin_data_dir_dev=$DE_OUT_DIR/$DE_CKPT_NUM/dev_ce_0_triviaqa.json \
    --warmup_steps=1000 \
    --logging_steps=100 \
    --save_steps=1000 \
    --max_steps=10000 \
    --log_dir=$TB_DIR \
    --fp16