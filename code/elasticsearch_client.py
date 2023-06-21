from elasticsearch import Elasticsearch, NotFoundError
import json
import re
from tqdm import tqdm
import string


class ElasticsearchClient:
    def __init__(self, host='localhost', port=9200):
        self.client = Elasticsearch([{'host': host, 'port': port}], timeout=30, max_retries=10, retry_on_timeout=True)
        print(self.client.info())


    def create_index(self, index_name, settings=None):
        with open('../data/settings.json', "r") as f:
            settings = json.load(f)

        if not settings:
            raise Exception("settings.json이 필요합니다.")

        if not self.client.indices.exists(index_name):
            self.client.indices.create(index=index_name, body=settings)
            print(f"Index {index_name} successfully created.")
        else:
            print(f"Index {index_name} already exists.")


    def delete_index(self, index_name):
        try:
            self.client.indices.delete(index=index_name)
            print(f"Index {index_name} successfully deleted.")
        except NotFoundError:
            print(f"Index {index_name} doesn't exist.")


    def get_indices(self):
        return self.client.indices.get_alias("*")


    def get_index_settings(self, index_name):
        try:
            return self.client.indices.get_settings(index_name)
        except NotFoundError:
            print(f"Index {index_name} doesn't exist.")


    def update_index_settings(self, index_name, new_settings):
        try:
            self.client.indices.put_settings(index=index_name, body=new_settings)
            print(f"Settings for index {index_name} successfully updated.")
        except NotFoundError:
            print(f"Index {index_name} doesn't exist.")


    def preprocess(self, text):
        def remove_(text):
            """ 정규표현식을 사용하여 불필요한 기호 제거 """
            text = re.sub("'", " ", text)
            text = re.sub('"', " ", text)
            text = re.sub("《", " ", text)
            text = re.sub("》", " ", text)
            text = re.sub("<", " ", text)
            text = re.sub(">", " ", text)
            text = re.sub("〈", " ", text)
            text = re.sub("〉", " ", text)
            text = re.sub("\(", " ", text)
            text = re.sub("\)", " ", text)
            text = re.sub("‘", " ", text)
            text = re.sub("’", " ", text)
            return text


        def white_space_fix(text: str) -> str:
            return " ".join(text.split())


        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)


        def lower(text: str) -> str:
            return text.lower()


        return white_space_fix(remove_punc(lower(remove_(text))))
    
    
    def load_data(self, dataset_path= "../data/wikipedia_documents.json"):
        with open(dataset_path, "r") as f:
            wiki = json.load(f)

        wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        wiki_texts = [self.preprocess(text) for text in wiki_texts]
        wiki_corpus = [{"document_text": wiki_texts[i]} for i in range(len(wiki_texts))]
        return wiki_corpus
    
    
    def insert_data(self, index_name):
        wiki_corpus = self.load_data("../data/wikipedia_documents.json")
        for i, text in enumerate(tqdm(wiki_corpus)):
            try:
                self.client.index(index=index_name, id=i, body=text)
            except:
                print(f"Unable to load document {i}.")

        print(f"Succesfully loaded {self.client.count(index=index_name)['count']} into {index_name}")
        
        
    def client_search(self, index_name, question, top_k, analyzer="standard"):
        question = question
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "document_text": {
                                    "query": question,
                                    "analyzer": analyzer
                                }
                            }
                        }
                    ]
                }
            }
        }
        response = self.client.search(index=index_name, body=query, size=top_k)
        return response
    
    
    def client_msearch(self, index_name, queries, top_k, analyzer="standard"):
        query_list = []
        
        for i in range(len(queries)):
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "document_text": {
                                        "query": queries[i],
                                        "analyzer": analyzer
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": top_k
            }
            query_list.append({"index": index_name})
            query_list.append(query)
            
        response = self.client.msearch(body=query_list)
        return response
    
    
def main():
    index_name = 'wiki_documents'
    es_client = ElasticsearchClient()
    es_client.delete_index(index_name) #> 처음실행시, index가 없기 때문에 404오류가 뜸 -> 두번째 실행때 주석 해제하고 사용
    es_client.create_index(index_name) #> 오류 if not self.client.indices.exists(index_name):
    settings = es_client.get_index_settings(index_name)
    print(settings)
    es_client.insert_data(index_name)


if __name__ == "__main__":
    main()