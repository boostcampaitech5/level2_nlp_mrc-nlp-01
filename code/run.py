import os
from datetime import datetime
from pytz import timezone
from inference import inference
from train import train_reader
from omegaconf import OmegaConf

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader, DensePassageRetriever, DenseRetriever, TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import SentenceTransformersRanker, TransformersSummarizer
import logging


logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


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
    document_store = ElasticsearchDocumentStore(
        return_embedding=True,
        analyzer="standard",         #> 새로운 Elasticsearch 인덱스를 생성할 때 빌트인 중 하나에서 기본 분석기를 지정합니다.
    )

    ## retriever
    retriever = BM25Retriever(document_store=document_store)
    # retriever = TfidfRetriever(document_store=document_store)

    ## reader
    reader = FARMReader(
        model_name_or_path=config.reader.model_name_or_path,
        progress_bar=True,
        context_window_size=150,                  #> 답변 주변의 컨텍스트를 표시할 때 사용되는 답변 범위 주변의 창 크기(문자)입니다.
        batch_size=config.reader.batch_size,      #> 모델이 추론을 위해 한 배치에서 받는 샘플 수입니다.
        use_gpu=True,
        no_ans_boost=0.0,                         #> no_answer 로짓이 얼마나 부스트/증가되었는지. 음수인 경우 "no_answer"가 예측될 가능성이 낮습니다. 양수이면 "no_answer"일 확률이 높아집니다.
        return_no_answer=False,                   #> 결과에 no_answer 예측을 포함할지 여부입니다.
        top_k=4,                                 #> 반환할 최대 답변 수
        max_seq_len=config.reader.max_seq_len,    #> 모델에 대한 하나의 입력 텍스트의 최대 시퀀스 길이. 초과 시 모든 것이 잘립니다.
        doc_stride=config.reader.doc_stride,      #> 긴 텍스트를 분할하기 위한 스트라이딩 윈도우의 길이
        duplicate_filtering=0,                    #> 0은 정확한 중복에 해당합니다. -1은 중복 제거를 끕니다. 답변의 시작 위치와 끝 위치가 모두 고려되며 위치에 따라 필터링됩니다.
        use_confidence_scores=True,               #> 예상 답변의 순위를 매기는 데 사용되는 점수 유형을 결정합니다. => [0, 1] 사이의 척도화된 신뢰도/관련성 점수.
        confidence_threshold=None,                #> confidence_threshold 미만의 예측을 필터링합니다. 값은 0과 1 사이여야 합니다.
        max_query_length=64                       #> 질문의 최대 길이.
        )

    if config.reader.train:
        reader = train_reader(config, reader)
    
    # ## ranker, summarizer
    # ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=10)
    # summarizer = TransformersSummarizer(model_name_or_path='t5-large', min_length=10, max_length=100)
    ## pipeline
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever) 
    # pipeline.add_node(component=ranker, name='Ranker', inputs=['Retriever'])
    # pipeline.add_node(component=summarizer, name='Summarizer', inputs=['Ranker'])
    
    inference(config.path, pipeline)
    

    ## 파이프라인 평가 결과 저장
    eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)
    eval_result = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": config.retriever.top_k}, "Reader": {"top_k": config.reader.top_k}})

    retriever_result = eval_result["Retriever"]
    reader_result = eval_result["Reader"]

    eval_result.save(f"../output/{config.path.output_dir}/pipeline_result") #> TO DO