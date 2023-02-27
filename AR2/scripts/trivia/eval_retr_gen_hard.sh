echo "start de model inference"

cd ../..

EXP_NAME=run_de_ict_triviaqa              # de means dual encoder.
DATA_DIR=./data/trivia
CKPT_NUM=40000
OUT_DIR=./output/$EXP_NAME

CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9539 \
    ./wiki/inference_de_wiki_gpu.py \
    --model_type="/srv/nfs/VESO/models/ernie-2.0-base-en" \
    --eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
    --output_dir=$OUT_DIR/$CKPT_NUM \
    --test_qa_path=$DATA_DIR/trivia-test.qa.csv \
    --train_qa_path=$DATA_DIR/trivia-train.qa.csv \
    --dev_qa_path=$DATA_DIR/trivia-dev.qa.csv \
    --max_seq_length=256 \
    --per_gpu_eval_batch_size=512 \
    --passage_path=$DATA_DIR/psgs_w100.tsv \
    --load_cache \
    --fp16