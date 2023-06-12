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
    save_dir = os.path.join(config.path.output_dir, config.reader.save_dir)
    
    train_csv_path = os.path.join(config.path.data_dir, config.path.train_csv)
    valid_csv_path = os.path.join(config.path.data_dir, config.path.valid_csv)
    
    train_for_reader = os.path.join(config.path.data_dir, config.path.train_for_reader)
    valid_for_reader = os.path.join(config.path.data_dir, config.path.valid_for_reader)
    
    convert_to_reader_train(train_csv_path, train_for_reader)
    convert_to_reader_train(valid_csv_path, valid_for_reader)
    
    ## train
    reader.train(
        data_dir=config.path.data_dir,
        use_gpu=True,
        batch_size=config.reader.batch_size,
        n_epochs=config.reader.n_epochs,
        learning_rate=config.reader.learning_rate,
        train_filename=train_for_reader,
        dev_filename=valid_for_reader,
        warmup_proportion=0.1,
        save_dir=save_dir
    )

    ## reader 평가
    eval_file = reader.eval_on_file(
        data_dir=config.path.data_dir,
        test_filename=config.path.dev_filename,
    )
    
    print(eval_file)