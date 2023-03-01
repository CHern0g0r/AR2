from typing import Tuple
from dataclasses import dataclass, field, fields


@dataclass
class MetaParamConfig:
    experiment_name: str = field(default=None)
    data_dir: str = field(default=None)
    out_dir: str = field(default=None)
    required_args: Tuple[str] = field(default=())

    @classmethod
    def fields(cls):
        return fields(cls)
    
    @classmethod
    def meta_fields(cls):
        return fields(MetaParamConfig)
    
    def __post_init__(self):
        if any(map(lambda x: getattr(self, x) is None, self.required_args)):
            raise ValueError('Required parameter missed')
        
    def __repr__(self):
        metafields = {
            k.name: getattr(self, k.name)
            for k in self.meta_fields()
        }
        flds = {
            k.name: getattr(self, k.name)
            for k in self.fields()
            if k.name not in metafields.keys()
        }
        metablock = '\n    '.join(
            ['Metafields:'] +
            list(map(
                lambda x: f'{x[0]} = {x[1]}',
                metafields.items()
            ))
        )
        block = '\n    '.join(
            ['Fields:'] +
            list(map(
                lambda x: f'{x[0]} = {x[1]}',
                flds.items()
            ))
        )
        return '\n'.join([
            self.__class__.__name__,
            metablock,
            block
        ])


@dataclass(repr=False)
class CommonParamConfig(MetaParamConfig):
    model_type: str = field(default=None)
    output_dir: str = field(default=None)
    max_seq_length: int = field(default=128)
    no_cuda: bool = field(default=False)
    fp16: bool = field(default=False)
    fp16_opt_level: str = field(default="O1")
    gradient_checkpointing: bool = field(default=False)
    local_rank: int = field(default=-1)
    server_ip: str = field(default="")
    server_port: str = field(default="")

    required_args: Tuple[str]= field(default=(
        'model_type',
        'output_dir'
    ))


@dataclass(repr=False)
class CommonTrainConfig(CommonParamConfig):
    model_name_or_path: str = field(default=None)
    num_epoch: int = field(default=0)
    config_name: str = field(default="")
    tokenizer_name: str = field(default="")
    max_query_length: int = field(default=32)
    triplet: bool = field(default=False)
    log_dir: str = field(default=None)
    optimizer: str = field(default="adamW")
    per_gpu_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-05)
    weight_decay: float = field(default=0.0)
    adam_epsilon: float = field(default=1e-08)
    max_grad_norm: float = field(default=2.0)
    max_steps: int = field(default=300000)
    warmup_steps: int = field(default=0)
    logging_steps: int = field(default=500)
    save_steps: int = field(default=500)
    seed: int = field(default=42)
    origin_data_dir: str = field(default=None)
    origin_data_dir_dev: str = field(default=None)
    load_optimizer_scheduler: bool = field(default=False)
    single_warmup: bool = field(default=True)
    adv_data_path: str = field(default=None)
    ann_data_path: str = field(default=None)
    number_neg: int = field(default=20)


@dataclass(repr=False)
class InferDEConfig(CommonParamConfig):
    eval_model_dir: str = field(default=None)
    test_qa_path: str = field(default=None)
    train_qa_path: str = field(default=None)
    dev_qa_path: str = field(default=None)
    passage_path: str = field(default=None)
    per_gpu_eval_batch_size: int = field(default=128)
    mode: str = field(default="train")
    load_cache: bool = field(default=False)
    top_k: int = field(default=100)
    thread_num: int = field(default=90)

    required_args: Tuple[str]= field(default=(
        "test_qa_path",
        "eval_model_dir",
        "passage_path"
    ))


@dataclass(repr=False)
class RunDEConfig(CommonTrainConfig):
    model_name_or_path_ict: str = field(default=None)
    cache_dir: str = field(default="")
    fix_embedding: bool = field(default=False)
    continue_train: bool = field(default=False)
    adv_loss_alpha: float = field(default=0.3)
    shuffle_positives: bool = field(default=False)
    reranker_model_path: str = field(default="")
    reranker_model_type: str = field(default="")
    adv_max_norm: float = field(default=0.0)
    adv_init_mag: float = field(default=0)
    adv_lr: float = field(default=0.05)
    adv_steps: int = field(default=3)


@dataclass(repr=False)
class RunCEConfig(CommonTrainConfig):
    data_dir: str = field(default=None)
    cache_dir: str = field(default="")
    contr_loss: bool = field(default=False)


@dataclass(repr=False)
class CoGenerateConfig(CommonTrainConfig):
    model_name_or_path_ict: str = field(default=None)
    corpus_path: str = field(default="")
    contr_loss: bool = field(default=False)
    normal_loss: bool = field(default=False)
    fix_embedding: bool = field(default=False)
    continue_train: bool = field(default=False)
    adv_loss_alpha: float = field(default=0.3)
    reranker_model_path: str = field(default="")
    reranker_model_type: str = field(default="")
    adv_lambda: float = field(default=0.0)
    adv_steps: int = field(default=3)
    test_qa_path: str = field(default="")
    train_qa_path: str = field(default="")
    dev_qa_path: str = field(default="")
    passage_path: str = field(default="")
    iteration_step: int = field(default=80)
    iteration_reranker_step: int = field(default=40)
    temperature_normal: float = field(default=1)
    scale_simmila: bool = field(default=False)
    reranker_learning_rate: float = field(default=0)
    load_cache: bool = field(default=False)
    ann_dir: str = field(default="")
    normal_term: str = field(default="cross_e")
    global_step: int = field(default=0)


@dataclass(repr=False)
class CoTrainConfig(CoGenerateConfig):
    normal_term: str = field(default="cross_e")
