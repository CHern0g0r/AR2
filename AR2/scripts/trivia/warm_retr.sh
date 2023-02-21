echo "start de model warmup"

cd ../..

mkdir output
mkdir tensorboard_log

EXP_NAME=run_de_ict_triviaqa   # de means dual encoder.
DATA_DIR=./data/trivia
Model_After_ICT=./checkpoints/wiki_ict.pt  # CKPT initialization, where we use ict-trained.  
OUT_DIR=../output/$EXP_NAME
TB_DIR=../tensorboard_log/$EXP_NAME    # tensorboard log path

if [[ -d $OUT_DIR ]]
then
    echo "out exists"
fi

pwd

if [[ -d $DATA_DIR ]]
then
    echo "data dir exists"
fi

if [[ -f $DATA_DIR/biencoder-trivia-train.json ]] & [[ -f $DATA_DIR/biencoder-trivia-dev.json ]]
then
    echo "data exists"
fi

CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9539 \
    ./wiki/run_de_model_ernie.py \
    --model_type="/srv/nfs/VESO/models/ernie-2.0-base-en" \
    --origin_data_dir=$DATA_DIR/biencoder-trivia-train.json \
    --origin_data_dir_dev=$DATA_DIR/biencoder-trivia-dev.json \
    --model_name_or_path_ict=$Model_After_ICT \
    --max_seq_length=256 \
    --per_gpu_train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-5 \
    --output_dir $OUT_DIR \
    --warmup_steps 4000 \
    --logging_steps 100 \
    --save_steps 1000 \
    --max_steps 40000 \
    --log_dir $TB_DIR \
    --fp16 \
    --number_neg 1