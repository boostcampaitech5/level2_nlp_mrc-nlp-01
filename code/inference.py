import os
import time
from datetime import datetime
from pytz import timezone
import pandas as pd
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from preprocessing import convert_to_reader_train
from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader, DensePassageRetriever, DenseRetriever, TfidfRetriever, DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline


def inference(config, pipe):

    ## 예측 실행
    test_path = os.path.join(config.data_dir, config.test_for_inference)
    test_data = pd.read_csv(test_path)

    preds= pipe.run_batch(
        queries=list(test_data['question'].values),
        params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 1}},
    )

    ## preds to output
    outputs = {}
    output_path = os.path.join(config.output_dir, 'outputs.json')
    for i in range(len(test_data)):
        outputs[test_data['id'][i]] = preds['answers'][i][0].answer
    with open(output_path, 'w', encoding='UTF-8') as f:
        json.dump(outputs, f, ensure_ascii=False)
        