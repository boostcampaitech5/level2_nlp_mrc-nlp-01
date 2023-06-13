import json
from tqdm import tqdm
import pyarrow as pa
from haystack import Label, Answer, Document
from haystack.document_stores import ElasticsearchDocumentStore
import re
from haystack.nodes import PreProcessor


def read_documents(path):
    """path에 있는 document를 읽어서 document형식으로 변환한 후 list에 저장한 뒤 반환합니다.

    Args:
        path: document파일의 path

    Returns:
        processed_tables: document들을 저장하고 있는 list
    """
    
    document_list = []
    
    with open(path) as tables:
        tables = json.load(tables)
        
    for key, table in tqdm(tables.items()):
        document = {
            "content": table.get('text', ''),
            "meta": {
                'corpus_source': table.get('corpus_source', ''),
                'url': table.get('url', ''),
                'domain': table.get('domain', ''), 
                'title': table.get('title', ''),
                'author': table.get('author', ''), 
                'html': table.get('html', ''), 
                'document_id': table.get('document_id', key)
            }
        }
        document = Document(content=document['content'], id=str(document['meta']['document_id']), meta=document['meta'])
        document_list.append(document)

    return document_list 


def clean_document_store(document_store):
    """document_store의 document, label을 모두 삭제합니다.

    Args:
        document_store
    """
    
    if len(document_store.get_all_documents()) or len(document_store.get_all_labels()) > 0:
        document_store.delete_documents("document")
        document_store.delete_documents("label")


def read_labels(valid_path):
    """path에 있는 정보들을 읽어서 label을 생성한 뒤, label의 list를 반환합니다.

    Args:
        valid_path: valid 데이터를 담고있는 파일의 path

    Returns:
        labels: label들을 저장하고 있는 list
    """
    
    ## Arrow 파일에서 데이터 스트림 생성
    with open(valid_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        valid_batches = [b for b in reader]

    ## Arrow RecordBatches를 Table로 병합 -> pandas DataFrame으로 변환
    df_valid = pa.Table.from_batches(valid_batches).to_pandas()
    
    labels = []
    for row in tqdm(df_valid.iterrows()):

        ## 필터링용 meta정보 사전
        meta = {"title": row["title"], "document_id": row["document_id"], "id": row["id"], "__index_level_0__": row["__index_level_0__"]} 

        ## 답이 있는 질문을 레이블에 추가
        if row['answers']['text']:
            for answer in tqdm(row['answers']['text']):
                label = Label(
                    query=row["question"], answer=Answer(answer=answer), origin="gold-label", document=Document(content=row["context"], id=row["id"]),
                    meta=meta, is_correct_answer=True, is_correct_document=True, no_answer=False)
                labels.append(label)

        ## 답이 없는 질문을 레이블에 추가
        else:
            label = Label(
                query=row["question"], answer=Answer(answer=""), origin="gold-label", document=Document(content=row["context"], id=row["id"]),
                meta=meta, is_correct_answer=True, is_correct_document=True, no_answer=True)
            labels.append(label)
        
    return labels


if __name__ == "__main__":
    """document_store에 document와 label을 추가합니다."""
    
    document_store = ElasticsearchDocumentStore(return_embedding=True, analyzer="standard")
    document_path = '../data/wikipedia_documents.json' # document를 저장하고 있는 파일
    
    ## Arrow 파일 경로
    valid_path = '../data/train_dataset/validation/dataset.arrow'

    ## document_store에 document 추가
    clean_document_store(document_store)
    document_list = read_documents(document_path)
    document_store.write_documents(document_list)
    print(f"{document_store.get_document_count()}개 문서가 저장되었습니다")
    
    ## document_store에 label 추가
    document_labels = read_labels(valid_path)
    document_store.write_labels(document_labels)
    print(f"{document_store.get_label_count()}개의 질문 답변 쌍을 로드했습니다.")
    
