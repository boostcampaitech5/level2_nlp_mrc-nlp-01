import os
from datetime import datetime
from pytz import timezone
from inference import inference
from train import train_reader
from omegaconf import OmegaConf

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader, DensePassageRetriever, DenseRetriever, TfidfRetriever, DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline


if __name__ == "__main__":
    
    ## config load
    config = OmegaConf.load('./config.yaml')

    ## 현제 시간 가져오기
    now = datetime.now(timezone('Asia/Seoul'))
    output_dir_name = now.strftime('%Y-%m-%d-%H:%M')
    print('output_dir_name : ', output_dir_name)
        
    ## output 폴더 생성
    os.makedirs(config.path.output_dir, exist_ok=True)
    config.path.output_dir = os.path.join(config.path.output_dir, output_dir_name)
    os.makedirs(config.path.output_dir, exist_ok=True)
    
    ## document_store
    document_store = ElasticsearchDocumentStore(return_embedding=True)

    ## retriever
    retriever = BM25Retriever(document_store)

    ## reader
    reader = FARMReader(
        model_name_or_path=config.reader.model_name_or_path,
        max_seq_len=config.reader.max_seq_len,
        doc_stride=config.reader.doc_stride,
        return_no_answer=False,
        progress_bar=True
    )

    if config.reader.train:
        reader = train_reader(config, reader)
    
    ## pipeline
    pipe = ExtractiveQAPipeline(reader, retriever)
    
    inference(config.path, pipe)
    