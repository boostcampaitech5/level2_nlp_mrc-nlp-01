from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelTrainingArguments(TrainingArguments):
    '''
    Model 학습에 필요한 Arguments들 정의
    '''
    num_train_epochs: int = field(
        default=5,
        metadata={
            "help": "number of train epochs"
        },
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={
            "help": "batch_size"
        },
    )
    learning_rate: float = field(
        default=9e-06,
        metadata={
            "help": "learning_rate"
        },
    )
    save_total_limit: int = field(
        default=3,
        metadata={
            "help": "save total limit"
        },
    )
    weight_decay: float = field(
        default=0.01,
        metadata={
            "help": "weight_decay"
        },
    )
    warmup_steps: int = field(
        default=500,
        metadata={
            "help": "warmup_steps"
        },
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={
            "help": "evaluation step"
        },
    )
    save_strategy: str = field(
        default="epoch",
        metadata={
            "help": "save_strategy, 바꾸고 싶으면 save_step도 함께 바꿔야 함"
        },
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": "load_best"
        },
    )
    metric_for_best_model: str = field(
        default='f1',
        metadata={
            "help": "metric (f1 / exact_match) 중 하나 입력"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='klue/roberta-large',  #> "klue/bert-base"
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=385,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=40,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )


@dataclass
class RetrievalArguments:
    """
    Argument to set which passage retriever to select among BM25, BM25_Plus, and BM25_L.
    """
    retrieval_name: Optional[str] = field(
        default="BM25",
        metadata={"help": "The name of the passage retriever."},
    )