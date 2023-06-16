import logging
import os
import sys

from arguments import DataTrainingArguments, ModelArguments, ModelTrainingArguments
from datasets import DatasetDict, load_from_disk
import evaluate
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
    EarlyStoppingCallback
)
from utils_qa import check_no_error, postprocess_qa_predictions
import torch
import wandb


torch.cuda.empty_cache()
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ModelTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    # logging 설정
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],)
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path,)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path, use_fast=True,)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config,)

    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(data_args: DataTrainingArguments, training_args: TrainingArguments, model_args: ModelArguments, datasets: DatasetDict, tokenizer, model,) -> None:
    
    column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    last_checkpoint, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

    def prepare_train_features(examples):
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
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while (token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
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

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    def post_processing_function(examples, features, predictions, training_args):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]

            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        '''
        early stopping을 사용하는 경우에는 compute_metrics의 return 값이 1개만 설정되어 있어야 함
        따라서 predictions의 EM, f1 2개 값 중 f1만 사용하도록 설정
        '''
        predictions = metric.compute(predictions=p.predictions, references=p.label_ids)
        wandb.log(predictions)
        return {'eval_exact_match': predictions['exact_match'], 'eval_f1' : predictions['f1']}


    # Trainer 초기화
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
    )  
    
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Training
    if training_args.do_train:
        torch.cuda.empty_cache()
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    wandb.login(key= 'your-API-key')
    wandb.init(project="MRC-Project")
    wandb.run.name = 'your-run-name'
    main()
