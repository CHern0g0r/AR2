from os.path import join
import sys

sys.path += ['../']
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from model.models import BiBertEncoder, HFBertEncoder, Reranker
from co_training_generate_new_train_wiki import RenewTools
from utils.lamb import Lamb
import random
from transformers import (
    AdamW,
    BertTokenizer,
)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
from utils.util import (
    set_seed,
    is_first_worker,
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    get_model_obj,
    CheckpointState,
    all_gather_list
)
import collections
retrieverBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "q_ids",
        "q_attn_mask",
        "c_ids",
        "c_attn_mask",
        "c_q_mapping",
        "is_positive",
    ],
)

def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0,
                  lr=0.0, eps=0.0) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if args.optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    elif args.optimizer == "lamb":
        return Lamb(optimizer_grouped_parameters, lr=lr, eps=eps)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))


def get_bert_reader_components(args, **kwargs):
    encoder = HFBertEncoder.init_encoder(
        args, model_type=args.reranker_model_type
    )
    hidden_size = encoder.config.hidden_size
    reranker = Reranker(encoder, hidden_size)

    return reranker

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path_ict",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--num_epoch",
        default=0,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--corpus_path",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=300000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--contr_loss",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--normal_loss",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--origin_data_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--origin_data_dir_dev",
        default=None,
        type=str,
    )
    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default=True,
        action="store_true",
        help="use single or re-warmup",
    )

    parser.add_argument("--adv_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )

    parser.add_argument("--ann_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )
    parser.add_argument(
        "--fix_embedding",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )
    parser.add_argument(
        "--continue_train",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )
    parser.add_argument(
        "--adv_loss_alpha",
        default=0.3,
        type=float,
        help="use single or re-warmup",
    )

    parser.add_argument("--reranker_model_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--reranker_model_type", type=str, default="", help="For distant debugging.")
    parser.add_argument("--number_neg", type=int, default=20, help="For distant debugging.")
    parser.add_argument("--adv_lambda", default=0., type=float)
    parser.add_argument("--adv_steps", default=3, type=int)
    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--test_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--train_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--dev_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--passage_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--iteration_step", default=80, type=int)
    parser.add_argument("--iteration_reranker_step", default=40, type=int)
    parser.add_argument("--temperature_normal", default=3, type=float)

    parser.add_argument("--scale_simmila",  default=False,  action="store_true")
    parser.add_argument("--reranker_learning_rate", default=0,type=float)
    parser.add_argument("--load_cache", default=False, action="store_true")
    parser.add_argument("--ann_dir", type=str, default="", help="For distant debugging.")

    parser.add_argument("--global_step", type=int, default=0, help="For distant debugging.")
    args = parser.parse_args()

    return args


def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def load_states_from_checkpoint_ict(model_file: str) -> CheckpointState:
    from torch.serialization import default_restore_location
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    new_stae_dict = {}
    for key, value in state_dict['model']['query_model']['language_model'].items():
        new_stae_dict['question_model.' + key] = value
    for key, value in state_dict['model']['context_model']['language_model'].items():
        new_stae_dict['ctx_model.' + key] = value
    return new_stae_dict


def load_model(args):
    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

     

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True)
    model = BiBertEncoder(args)
    if args.model_name_or_path_ict is not None:
        saved_state = load_states_from_checkpoint_ict(args.model_name_or_path_ict)
        model.load_state_dict(saved_state)
    if args.model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        model.load_state_dict(saved_state.model_dict)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    return tokenizer, model, None

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model

def get_new_dataset(args,model,global_step,renew_tools):
    model_path = os.path.join(args.output_dir, 'checkpoint-' + str(global_step))
    saved_state = load_states_from_checkpoint(model_path)
    model_to_load = get_model_obj(model)
    model_to_load.load_state_dict(saved_state.model_dict) 
    model.eval()
    logger.info(" model_path = %s", model_path)
    # model.to(args.device)
    with torch.no_grad():
        passage_embedding, passage_embedding_id = renew_tools.get_passage_embedding(args, model)
        torch.distributed.barrier()
        if is_first_worker():
            train_q,train_a,train_q_embed, train_q_embed2id = renew_tools.get_question_embedding(args,
                                                    model,args.train_qa_path,mode='train')
            dev_q,dev_a,dev_q_embed, dev_q_embed2id = renew_tools.get_question_embedding(args,
                                                    model,args.dev_qa_path,mode='dev')
            test_q,test_a,test_q_embed, test_q_embed2id = renew_tools.get_question_embedding(args,
                                                    model,args.test_qa_path,mode='test')

            gpu_index_flat, passage_embedding2id = renew_tools.get_new_faiss_index(args, passage_embedding,
                                                                                   passage_embedding_id)

            renew_tools.get_question_topk(train_q,train_a,train_q_embed, train_q_embed2id, args.origin_data_dir,
                                                 gpu_index_flat, passage_embedding2id,
                                                 mode='train', step_num=global_step)
            renew_tools.get_question_topk(dev_q,dev_a,dev_q_embed, dev_q_embed2id, args.origin_data_dir_dev,
                                                 gpu_index_flat, passage_embedding2id,
                                                 mode='dev', step_num=global_step)
            renew_tools.get_question_topk(test_q,test_a,test_q_embed, test_q_embed2id, args.origin_data_dir_dev,
                                                 gpu_index_flat, passage_embedding2id,
                                                 mode='test', step_num=global_step)

        # torch.distributed.barrier()
def eval_first(args,model,global_step,renew_tools):
    # model_path = os.path.join(args.output_dir, 'checkpoint-' + str(global_step))
    if args.model_name_or_path_ict is not None:
        saved_state = load_states_from_checkpoint_ict(args.model_name_or_path_ict)
        model.load_state_dict(saved_state)
    if args.model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        model.load_state_dict(saved_state.model_dict, strict=False)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        passage_embedding, passage_embedding_id = renew_tools.get_passage_embedding(args, model)
        torch.distributed.barrier()
        if is_first_worker():
            test_q,test_a,test_q_embed, test_q_embed2id = renew_tools.get_question_embedding(args,
                                                    model,args.test_qa_path,mode='test')

            gpu_index_flat, passage_embedding2id = renew_tools.get_new_faiss_index(args, passage_embedding,
                                                                                   passage_embedding_id)
            renew_tools.get_question_topk(test_q,test_a,test_q_embed, test_q_embed2id, args.origin_data_dir_dev,
                                                 gpu_index_flat, passage_embedding2id,
                                                 mode='test', step_num=global_step)

    # torch.distributed.barrier()
def main():
    args = get_arguments()
    set_env(args)
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)
    tokenizer, model,reranker_model = load_model(args)
    # load passage
    temp_slice_dir = os.path.join(args.ann_dir, 'temp')
    renew_tools = RenewTools(passages_path=args.passage_path, tokenizer=tokenizer,
                             output_dir=args.ann_dir, temp_dir=temp_slice_dir)
    # renew_tools = None
    dist.barrier()
    global_step = args.global_step
    if global_step > args.max_steps:
        pass
    else:
        get_new_dataset(args,model,global_step,renew_tools)
    # dist.barrier()
    logger.info(" global_step = %s", global_step)

    # if args.local_rank != -1:
    #     dist.barrier()


if __name__ == "__main__":
    main()
