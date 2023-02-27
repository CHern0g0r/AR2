cd ../..

EXP_NAME=run_de_ict_triviaqa
DATA_DIR=./data/trivia
CKPT_NUM=40000
OUT_DIR=./output/$EXP_NAME
TOPK_FILE=$OUT_DIR/$CKPT_NUM/dev_result_dict_list.json # dev_result_dict_list.json is generate in previous step
CE_TRAIN_FILE=$OUT_DIR/$CKPT_NUM/dev_ce_0_triviaqa.json
LABELED_FILE=$DATA_DIR/biencoder-trivia-dev.json

python ./utils/prepare_ce_data.py $TOPK_FILE $CE_TRAIN_FILE $LABELED_FILE   # generate dev set file


TOPK_FILE=$OUT_DIR/$CKPT_NUM/train_result_dict_list.json # train_result_dict_list.json is generate in previous step
CE_TRAIN_FILE=$OUT_DIR/$CKPT_NUM/train_ce_0_triviaqa.json
LABELED_FILE=$DATA_DIR/biencoder-trivia-train.json

python ./utils/prepare_ce_data.py $TOPK_FILE $CE_TRAIN_FILE $LABELED_FILE # generate train set file