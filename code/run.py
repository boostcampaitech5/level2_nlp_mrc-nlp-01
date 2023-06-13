import os
from datetime import datetime
from pytz import timezone
from inference import inference
from train import train_reader
from omegaconf import OmegaConf
import shutil

from transformers import set_seed
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader, TfidfRetriever, DensePassageRetriever, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import SentenceTransformersRanker, TransformersSummarizer
import logging


logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

if __name__ == "__main__":
    
    ## config load
    config = OmegaConf.load('./config.yaml')
    
    ## 시드 고정
    set_seed(42)
    
    ## 현제 시간 가져오기
    now = datetime.now(timezone('Asia/Seoul'))
    output_dir_name = now.strftime('%Y-%m-%d-%H:%M')
    print('output_dir_name : ', output_dir_name)
        
    ## output 폴더 생성
    os.makedirs(config.path.output_dir, exist_ok=True)
    config.path.output_dir = os.path.join(config.path.output_dir, output_dir_name)
    os.makedirs(config.path.output_dir, exist_ok=True)
    
    shutil.copyfile('./config.yaml', os.path.join(config.path.output_dir, 'config.yaml')) # yaml 파일 복사
        
    ## document_store init
    ## https://docs.haystack.deepset.ai/reference/document-store-api#elasticsearchdocumentstore__init__
    document_store = ElasticsearchDocumentStore(
        return_embedding=True
    )

    ## retriever init
    retriever = BM25Retriever(
        document_store=document_store,
        top_k=config.retriever.top_k
    )

    ## reader init
    reader = FARMReader(
        model_name_or_path=config.reader.init.model_name_or_path,
        progress_bar=True,
        context_window_size=config.reader.init.context_window_size,
        batch_size=config.reader.init.batch_size,
        use_gpu=True,
        no_ans_boost=config.reader.init.no_ans_boost,
        return_no_answer=False,
        top_k=config.reader.init.top_k,
        top_k_per_candidate=config.reader.init.top_k_per_candidate,
        top_k_per_sample=config.reader.init.top_k_per_sample,
        max_seq_len=config.reader.init.max_seq_len,
        doc_stride=config.reader.init.doc_stride,
        duplicate_filtering=config.reader.init.duplicate_filtering,
        use_confidence_scores=config.reader.init.use_confidence_scores,
        max_query_length=config.reader.init.max_query_length
    )

    if config.reader.is_train:
        reader = train_reader(config, reader)
    
    # ## ranker, summarizer
    # ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=10)
    # summarizer = TransformersSummarizer(model_name_or_path='t5-large', min_length=10, max_length=100)

    
    ## pipeline
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever) 
    # pipeline.add_node(component=ranker, name='Ranker', inputs=['Retriever'])
    # pipeline.add_node(component=summarizer, name='Summarizer', inputs=['Ranker'])
    
    ## inference
    inference(config, pipeline)
    
    ## 파이프라인 평가 결과 저장
    eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)
    eval_result = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": config.retriever.top_k}, "Reader": {"top_k": config.reader.top_k}})

    retriever_result = eval_result["Retriever"]
    reader_result = eval_result["Reader"]

    eval_result.save(os.path.join(config.path.output_dir, 'eval_pipeline_result'))
