"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
from arguments import DataTrainingArguments, ModelArguments, RetrievalArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk
)
import evaluate
from retrieval import SparseRetrieval, BM25, BM25_Plus, BM25_L, ElasticsearchRetrieval, Retriever_Ensemble
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, RetrievalArguments))
    model_args, data_args, training_args, retrieval_args = parser.parse_args_into_dataclasses()
    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],)

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path,)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path, use_fast=True,)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config,)

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(tokenizer.tokenize, datasets, training_args, data_args, retrieval_args,)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(tokenize_fn: Callable[[str], List[str]], datasets: DatasetDict, training_args: TrainingArguments, data_args: DataTrainingArguments, retrieval_args: RetrievalArguments, data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",) -> DatasetDict:
    retrieval_name = retrieval_args.retrieval_name

    if retrieval_name == "SparseRetrieval":
        retriever = SparseRetrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
    elif retrieval_name == "BM25":
        retriever = BM25(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
    elif retrieval_name == "BM25_Plus":
        retriever = BM25_Plus(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
    elif retrieval_name == "BM25_L":
        retriever = BM25_L(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
    elif retrieval_name == "Elasticsearch":
        retriever = ElasticsearchRetrieval()
    elif retrieval_name == "Ensemble":
        retriever = Retriever_Ensemble(tokenize_fn=tokenize_fn, datasets=datasets, topk=data_args.top_k_retrieval, data_path=data_path, context_path="wikipedia_documents.json",)
        df = retriever.ensemble_and_rerank()
    else:
        raise ValueError("SparseRetrieval, BM25, BM25_Plus, BM25_L, Elasticsearch, Ensemble 중 하나를 정확히 입력해 주세요.")
    
    if retrieval_name != "Ensemble" and retrieval_name != "Elasticsearch":
        retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    elif retrieval_name == "Ensemble":
        print("Retriever Ensemble")
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context_id": Sequence(Value(dtype="int64", id=None)),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context_id": Sequence(Value(dtype="int64", id=None)),
            }
        )
    if retrieval_name == "SparseRetrieval":
        del f["context_id"]   
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(data_args: DataTrainingArguments, training_args: TrainingArguments, model_args: ModelArguments, datasets: DatasetDict, tokenizer, model,) -> None:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [(o if sequence_ids[k] == context_index else None) for k, o in enumerate(tokenized_examples["offset_mapping"][i])]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Post-processing:
    def post_processing_function(examples, features, predictions: Tuple[np.ndarray, np.ndarray], training_args: TrainingArguments,) -> EvalPrediction:
        predictions = postprocess_qa_predictions(examples=examples, features=features, predictions=predictions, max_answer_length=data_args.max_answer_length, output_dir=training_args.output_dir,)
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]

            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=eval_dataset, test_examples=datasets["validation"])

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
