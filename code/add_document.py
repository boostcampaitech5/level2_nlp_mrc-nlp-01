import json
from tqdm import tqdm
import pyarrow as pa
from haystack import Label, Answer, Document
from haystack.document_stores import ElasticsearchDocumentStore
import re
from haystack.nodes import PreProcessor

## elasticsearch를 local로 실행하고 코드를 진행해 주세요

## document_store 설정 -> 기본으로 localhost, 9200 port가 부여됩니다
document_store = ElasticsearchDocumentStore(return_embedding=True, analyzer="standard",)


# 저장소 비우기
if len(document_store.get_all_documents()) or len(document_store.get_all_labels()) > 0:
    document_store.delete_documents("document")
    document_store.delete_documents("label")

def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\u000A", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”:%&\《\》〈〉''㈜·\-\'+\s一-龥サマーン]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

preprocessor = PreProcessor(
    split_by="word", #> sentence
    split_length=300,
    split_overlap=0,
    split_respect_sentence_boundary=True,
    clean_empty_lines=False,
    clean_whitespace=False,
    max_chars_check= 12000,
)

filename = '../data/wikipedia_documents.json'

def read_tables(filename):
    processed_tables = []
    with open(filename) as tables:
        tables = json.load(tables)
        for key, table in tables.items():
            document = {
                "content": table.get('text', ''),
                # "content": preprocess(table.get('text', '')),
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
    
    docs = preprocessor.process(processed_tables)
    return docs

tables = read_tables(filename)
print(tables[1].content)
print(f'document_store에 문서 저장중...약 5분 정도 소요됩니다.')
document_store.write_documents(tables)
print(f"{document_store.get_document_count()}개 문서가 저장되었습니다")



# Arrow 파일 경로
train_file_path = '../data/train_dataset/train/dataset.arrow'
valid_file_path = '../data/train_dataset/validation/dataset.arrow'
test_file_path = '../data/test_dataset/validation/dataset.arrow'

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
    if row['answers']['text']:
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