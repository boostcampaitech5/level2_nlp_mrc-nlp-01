import json
from tqdm import tqdm
import pyarrow as pa
from haystack import Label, Answer, Document
from haystack.document_stores import ElasticsearchDocumentStore

## elasticsearch를 local로 실행하고 코드를 진행해 주세요

## document_store 설정 -> 기본으로 localhost, 9200 port가 부여됩니다
document_store = ElasticsearchDocumentStore(return_embedding=True, analyzer="standard",)

filename = '/opt/ml/input/data/wikipedia_documents.json'

def read_tables(filename):
    processed_tables = []
    with open(filename) as tables:
        tables = json.load(tables)
        for key, table in tables.items():
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
            doc = Document(content=document['content'], id=str(document['meta']['document_id']), meta=document['meta'])
            processed_tables.append(doc)

    return processed_tables 

tables = read_tables(filename)
document_store.write_documents(tables)
print(f"{document_store.get_document_count()}개 문서가 저장되었습니다")



# Arrow 파일 경로
train_file_path = '/opt/ml/input/data/train_dataset/train/dataset.arrow'
valid_file_path = '/opt/ml/input/data/train_dataset/validation/dataset.arrow'
test_file_path = '/opt/ml/input/data/test_dataset/validation/dataset.arrow'

# Arrow 파일에서 데이터 스트림 생성
with open(train_file_path, 'rb') as f:
    reader = pa.ipc.open_stream(f)
    train_batches = [b for b in reader]

with open(valid_file_path, 'rb') as f:
    reader = pa.ipc.open_stream(f)
    valid_batches = [b for b in reader]

with open(test_file_path, 'rb') as f:
    reader = pa.ipc.open_stream(f)
    test_batches = [b for b in reader]

# Arrow RecordBatches를 Table로 병합
train_table = pa.Table.from_batches(train_batches)
valid_table = pa.Table.from_batches(valid_batches)
test_table = pa.Table.from_batches(test_batches)

# Arrow Table을 pandas DataFrame으로 변환
df_train = train_table.to_pandas()
df_val = valid_table.to_pandas()
df_test = test_table.to_pandas()

dfs = {'train': df_train, 'validation': df_val}

labels = []
for i, row in dfs["validation"].iterrows():

    meta = {"title": row["title"], "document_id": row["document_id"], "id": row["id"], "__index_level_0__": row["__index_level_0__"]} #> 필터링용

    # 답이 있는 질문을 레이블에 추가
    if len(row['answers']['text']):
        for answer in row['answers']['text']:
            label = Label(
                query=row["question"], answer=Answer(answer=answer), origin="gold-label", document=Document(content=row["context"], id=row["id"]),
                meta=meta, is_correct_answer=True, is_correct_document=True, no_answer=False)
            labels.append(label)

    # 답이 없는 질문을 레이블에 추가
    else:
        label = Label(
            query=row["question"], answer=Answer(answer=""), origin="gold-label", document=Document(content=row["context"], id=row["id"]),
            meta=meta, is_correct_answer=True, is_correct_document=True, no_answer=True)
        labels.append(label)

document_store.write_labels(labels)
print(f"""{document_store.get_label_count()}개의 질문 답변 쌍을 로드했습니다.""")