import os
import time
from datetime import datetime
from pytz import timezone
import pandas as pd
import json
from tqdm import tqdm
from preprocessing import convert_to_reader_train
from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader, DensePassageRetriever, DenseRetriever, TfidfRetriever, DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline

def train_reader(config, reader):
    """FARM_reader를 훈련합니다

    Args:
        config: config.path 와 config.reader.train 부분만 사용합니다
        reader: 훈련할 reader

    Returns:
        reader: 훈련된 reader
    """
    save_dir = os.path.join(config.path.output_dir, config.reader.train.save_dir)
    
    train_csv_path = os.path.join(config.path.data_dir, config.path.train_csv)
    valid_csv_path = os.path.join(config.path.data_dir, config.path.valid_csv)
    
    train_for_reader = os.path.join(config.path.data_dir, config.path.train_for_reader)
    valid_for_reader = os.path.join(config.path.data_dir, config.path.valid_for_reader)
    
    ## train을 위해서 squard형식으로 데이터를 변경합니다. 결과로 .json파일이 생성됩니다.
    convert_to_reader_train(train_csv_path, train_for_reader)
    convert_to_reader_train(valid_csv_path, valid_for_reader)
    
    ## train
    reader.train(
        data_dir=config.path.data_dir,
        use_gpu=True,
        batch_size=config.reader.train.batch_size,
        n_epochs=config.reader.train.n_epochs,
        learning_rate=config.reader.train.learning_rate,
        train_filename=train_for_reader,
        dev_filename=valid_for_reader,
        warmup_proportion=config.reader.train.warmup_proportion,
        save_dir=save_dir,
        # early_stopping=early_stopping,
    )

    ## validation set을 가지고 reader를 평가합니다
    eval_file = reader.eval_on_file(
        data_dir=config.path.data_dir,
        test_filename=valid_for_reader,
    )
    
    print(eval_file)
    
    return reader