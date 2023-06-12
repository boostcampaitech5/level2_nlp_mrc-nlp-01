import json
from tqdm import tqdm
from haystack.document_stores import ElasticsearchDocumentStore

## elasticsearch를 local로 실행하고 코드를 진행해 주세요

## document_store 설정 -> 기본으로 localhost, 9200 port가 부여됩니다
document_store = ElasticsearchDocumentStore(return_embedding=True)

with open('../data/wikipedia_documents.json') as f:
    wiki = json.load(f)

## local elasticsearch에서 이제까지 저장된 모든 document들을 불러옵니다
count = document_store.get_document_count() - 1

for i in tqdm(range(count, len(wiki))):
    document = [{
        "content": wiki[str(i)]['text'],
        "meta": {
            'corpus_source': wiki[str(i)]['corpus_source'],
            'url': wiki[str(i)]['url'],
            'domain': wiki[str(i)]['domain'], 
            'title': wiki[str(i)]['title'],
            'author': wiki[str(i)]['author'], 
            'html': wiki[str(i)]['html'], 
            'document_id': wiki[str(i)]['document_id']
        }
    }]
    ## 기본적으로, 중복되는 document들은 overwrite되기 때문에 상관 없습니다.
    document_store.write_documents(document)