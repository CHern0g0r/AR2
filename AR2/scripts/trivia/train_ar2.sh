echo "start co-training"
EXP_NAME=co_training_triviaqa
TB_DIR=../tensorboard_log/$EXP_NAME    # tensorboard log path
OUT_DIR=../output/$EXP_NAME

DE_EXP_NAME=run_de_ict_triviaqa
CE_EXP_NAME=run_ce_model
DE_CKPT_PATH=../output/$DE_EXP_NAME/checkpoint-40000
CE_CKPT_PATH=../output/$CE_EXP_NAME/checkpoint-4000
Origin_Data_Dir=../output/$DE_EXP_NAME/40000/train_ce_0_triviaqa.json
Origin_Data_Dir_Dev=../output/$DE_EXP_NAME/40000/dev_ce_0_triviaqa.json

Reranker_TYPE=nghuyong/ernie-2.0-large-en
Iteration_step=2000 
Iteration_reranker_step=500
MAX_STEPS=32000

# for global_step in `seq 0 2000 $MAX_STEPS`; do echo $global_step; done;
for global_step in `seq 0 $Iteration_step $MAX_STEPS`; 
do 
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 $BASE_SCRIPT_DIR/wiki/co_training_wiki_train.py \
        --model_type=nghuyong/ernie-2.0-en \
        --model_name_or_path=$DE_CKPT_PATH \
        --max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
        --number_neg=15 --learning_rate=1e-5 \
        --reranker_model_type=$Reranker_TYPE \
        --reranker_model_path=$CE_CKPT_PATH \
        --reranker_learning_rate=1e-6 \
        --output_dir=$OUT_DIR \
        --log_dir=$TB_DIR \
        --origin_data_dir=$Origin_Data_Dir \
        --origin_data_dir_dev=$Origin_Data_Dir_Dev \
        --warmup_steps=2000 --logging_steps=10 --save_steps=2000 --max_steps=$MAX_STEPS \
        --gradient_checkpointing --normal_loss \
        --iteration_step=$Iteration_step \
        --iteration_reranker_step=$Iteration_reranker_step \
        --temperature_normal=1 --ann_dir=$OUT_DIR/temp --adv_lambda 0.5 --global_step=$global_step

    g_global_step=`expr $global_step + $Iteration_step`
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 $BASE_SCRIPT_DIR/wiki/co_training_wiki_generate.py \
        --model_type=nghuyong/ernie-2.0-en \
        --model_name_or_path=$Warm_de_path \
        --max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
        --number_neg=15 --learning_rate=1e-5 \
        --reranker_model_type=$Reranker_TYPE \
        --reranker_model_path=$Warm_Reranker_PATH \
        --reranker_learning_rate=1e-6 \
        --output_dir=$BASE_DIR/ckpt/$EXP_NAME \
        --log_dir=tensorboard/logs/$EXP_NAME \
        --origin_data_dir=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_ce_0_triviaqa.json \
        --origin_data_dir_dev=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_ce_0_triviaqa.json \
        --train_qa_path=$BASE_DIR/data/trivia_data/trivia-train.qa.csv \
        --test_qa_path=$BASE_DIR/data/trivia_data/trivia-test.qa.csv \
        --dev_qa_path=$BASE_DIR/data/trivia_data/trivia-dev.qa.csv \
        --passage_path=$BASE_DIR/data/psgs_w100.tsv \
        --warmup_steps=2000 --logging_steps=10 --save_steps=2000 --max_steps=$MAX_STEPS \
        --gradient_checkpointing --normal_loss --adv_step=0 \
        --iteration_step=$Iteration_step \
        --iteration_reranker_step=$Iteration_reranker_step \
        --temperature_normal=1 --ann_dir=$BASE_DIR/ckpt/$EXP_NAME/temp --adv_lambda=0.5 --global_step=$g_global_step
done
