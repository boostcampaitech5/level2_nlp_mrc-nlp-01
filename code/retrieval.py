import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi, BM25Plus, BM25L
from elasticsearch_client import ElasticsearchClient
from sklearn.preprocessing import MinMaxScaler


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class Retrieval:
    def __init__(self, tokenize_fn, data_path: Optional[str] = "../data/", context_path: Optional[str] = "wikipedia_documents.json",):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.tokenize_fn = tokenize_fn

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                if __name__ == "__main__":
                    tmp["scores"] = doc_scores[idx]
                    if tmp["original_context"] in tmp["context"]:
                        tmp["answer_index"] = [self.contexts[pid] for pid in doc_indices[idx]].index(tmp["original_context"])
                    else:
                        tmp["answer_index"] = -1
                
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    def get_sparse_embedding(self):
        pass
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        pass
    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        pass


class BM25(Retrieval):
    def __init__(self, tokenize_fn, data_path: Optional[str] = "../data/",  context_path: Optional[str] = "wikipedia_documents.json"):
        super().__init__(tokenize_fn, data_path, context_path)
        self.bm25 = None
        
    def get_sparse_embedding(self):
        with timer("bm25 building"):
            self.bm25 = BM25Okapi(self.contexts, tokenizer=self.tokenize_fn) 
        
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_query = self.tokenize_fn(query)
        with timer("query ex search"):
            result = self.bm25.get_scores(tokenized_query)
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_queris = [self.tokenize_fn(query) for query in queries]
        with timer("query ex search"):
            result = np.array([self.bm25.get_scores(tokenized_query) for tokenized_query in tqdm(tokenized_queris)])
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class BM25_Plus(Retrieval):
    def __init__(self, tokenize_fn, data_path: Optional[str] = "../data/",  context_path: Optional[str] = "wikipedia_documents.json"):
        super().__init__(tokenize_fn, data_path, context_path)
        self.bm25_plus = None
    def get_sparse_embedding(self):
        with timer("bm25_plus building"):
            self.bm25_plus = BM25Plus(self.contexts, tokenizer=self.tokenize_fn) 
    
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_query = self.tokenize_fn(query)
        with timer("query ex search"):
            result = self.bm25_plus.get_scores(tokenized_query)
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_queris = [self.tokenize_fn(query) for query in queries]
        with timer("query ex search"):
            result = np.array([self.bm25_plus.get_scores(tokenized_query) for tokenized_query in tqdm(tokenized_queris)])
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class BM25_L(Retrieval):
    def __init__(self, tokenize_fn, data_path: Optional[str] = "../data/",  context_path: Optional[str] = "wikipedia_documents.json"):
        super().__init__(tokenize_fn, data_path, context_path)
        self.bm25_l = None
    def get_sparse_embedding(self):
        with timer("bm25_l building"):
            self.bm25_l = BM25L(self.contexts, tokenizer=self.tokenize_fn) 
    
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_query = self.tokenize_fn(query)
        with timer("query ex search"):
            result = self.bm25_l.get_scores(tokenized_query)
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_queris = [self.tokenize_fn(query) for query in queries]
        with timer("query ex search"):
            result = np.array([self.bm25_l.get_scores(tokenized_query) for tokenized_query in tqdm(tokenized_queris)])
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class ElasticsearchRetrieval:
    def __init__(self):
        self.client = ElasticsearchClient()
        self.index_name = 'wiki_documents'
        self.analyzer = "standard"
        
    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, doc = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc[i])

            return (doc_scores, [[doc[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices, doc = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(d for d in doc[idx]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                if __name__ == "__main__":
                    tmp["scores"] = doc_scores[idx]
                    if tmp["original_context"] in tmp["context"]:
                        tmp["answer_index"] = [self.contexts[pid] for pid in doc_indices[idx]].index(tmp["original_context"])
                    else:
                        tmp["answer_index"] = -1
                
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc(self, query: str, k: Optional[int] = 1):
        ## elasticsearch 검색
        response = self.client.client_search(self.index_name, query, top_k=k, analyzer=self.analyzer)
        
        ## 검색결과를 doc, doc_score로 변환
        doc = [response['hits']['hits'][i]['_source']['document_text'] for i in range(len(response['hits']))]
        doc_indices = [response['hits']['hits'][i]['_id']['document_text'] for i in range(len(response['hits']))]
        doc_scores = [response['hits']['hits'][i]['_score'] for i in range(len(response['hits']))]
        
        return doc_scores, doc_indices, doc
    
    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1):
        ## elasticsearch 검색
        response = self.client.client_msearch(self.index_name, queries, top_k=k, analyzer=self.analyzer)
        
        ## 검색결과를 doc, doc_score로 변환
        doc = [[response['responses'][i]['hits']['hits'][j]['_source']['document_text'] for j in range(len(response['responses'][i]['hits']))] for i in range(len(response['responses']))]
        doc_indices = [[response['responses'][i]['hits']['hits'][j]['_id'] for j in range(len(response['responses'][i]['hits']))] for i in range(len(response['responses']))]
        doc_scores = [[response['responses'][i]['hits']['hits'][j]['_score'] for j in range(len(response['responses'][i]['hits']))] for i in range(len(response['responses']))]
        
        return doc_scores, doc_indices, doc
        
        
class SparseRetrieval:
    def __init__(self, tokenize_fn, data_path: Optional[str] = "../data/", context_path: Optional[str] = "wikipedia_documents.json",) -> None:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.tfidfv = TfidfVectorizer(tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000,)
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> None:
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> None:
        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                if __name__ == "__main__":
                    tmp["scores"] = doc_scores[idx]
                    if tmp["original_context"] in tmp["context"]:
                        tmp["answer_index"] = [self.contexts[pid] for pid in doc_indices[idx]].index(tmp["original_context"])
                    else:
                        tmp["answer_index"] = -1                     
                    
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (np.sum(query_vec) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.tfidfv.transform(queries)
        assert (np.sum(query_vec) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                if __name__ == "__main__":
                    tmp["scores"] = doc_scores[idx]
                    if tmp["original_context"] in tmp["context"]:
                        tmp["answer_index"] = [self.contexts[pid] for pid in doc_indices[idx]].index(tmp["original_context"])
                    else:
                        tmp["answer_index"] = -1
                
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.tfidfv.transform([query])
        assert (np.sum(query_vec) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vecs = self.tfidfv.transform(queries)
        assert (np.sum(query_vecs) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


class Retriever_Ensemble(Retrieval):
    def __init__(self, tokenize_fn: Callable[[str], List[str]], topk: int, data_path: str = "../data", context_path: str = "wikipedia_documents.json", datasets=None,):
        self.tokenize_fn = tokenize_fn
        self.topk = topk
        self.data_path = data_path
        self.context_path = context_path        
        
        if __name__ == "__main__":
            self.datasets = datasets
        else:
            self.datasets = datasets["validation"]
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로

        self.ids = list(range(len(self.contexts)))
        self.tf_idf = SparseRetrieval(tokenize_fn=tokenize_fn, data_path=self.data_path, context_path=self.context_path)
        self.bm25 = BM25(tokenize_fn=tokenize_fn, data_path=self.data_path, context_path=self.context_path)
        self.bm25_plus = BM25_Plus(tokenize_fn=tokenize_fn, data_path=self.data_path, context_path=self.context_path)

        self.tf_idf.get_sparse_embedding()
        self.bm25.get_sparse_embedding()
        self.bm25_plus.get_sparse_embedding()

        if isinstance(self.datasets, Dataset):
            self.length_query_or_dataset = len(self.datasets)
            self.tf_idf_scores, self.tf_idf_indices = self.tf_idf.get_relevant_doc_bulk(self.datasets["question"], k=self.topk)
            self.bm25_scores, self.bm25_indices = self.bm25.get_relevant_doc_bulk(self.datasets["question"], k=self.topk)
            self.bm25_plus_scores, self.bm25_plus_indices = self.bm25_plus.get_relevant_doc_bulk(self.datasets["question"], k=self.topk)  
        elif isinstance(self.datasets, str):
            self.length_query_or_dataset = 1
            self.tf_idf_scores, self.tf_idf_indices = self.tf_idf.get_relevant_doc(self.datasets, k=self.topk)
            self.bm25_scores, self.bm25_indices = self.bm25.get_relevant_doc(self.datasets, k=self.topk)
            self.bm25_plus_scores, self.bm25_plus_indices = self.bm25_plus.get_relevant_doc(self.datasets, k=self.topk)    
    
    def ensemble_and_rerank(self):
        '''추출된 TF-IDF, BM25, BM25+ 각각의 top-k passage들의 score를 Min-Max 정규화하여 모두 합친 뒤, 다시 top-k를 선정하여 retrieve된 passage들을 pd.DataFrame으로 반환합니다'''
        all_selected_scores, all_selected_indices = [], []
        
        if isinstance(self.datasets, str):
            self.tf_idf_scores = [self.tf_idf_scores]
            self.bm25_scores = [self.bm25_scores]
            self.bm25_plus_score = [self.bm25_plus_scores]
            
            self.tf_idf_indices = [self.tf_idf_indices]
            self.bm25_indices = [self.bm25_indices]
            self.bm25_plus_indices = [self.bm25_plus_indices]
        # 전체 test_data에 대한 각각의 top-k passage들의 score에 min max scaling를 적용시킵니다. 
        for i in range(self.length_query_or_dataset):
            tf_idf_scores = np.array(self.tf_idf_scores[i],float).reshape(-1, 1)
            bm25_scores = np.array(self.bm25_scores[i],float).reshape(-1, 1)
            bm25_plus_scores = np.array(self.bm25_plus_scores[i],float).reshape(-1, 1)

            scaler_tf_idf = MinMaxScaler()
            scaler_bm25 = MinMaxScaler()
            scaler_bm25_plus = MinMaxScaler()

            scaler_tf_idf.fit(tf_idf_scores)
            scaler_bm25.fit(bm25_scores)
            scaler_bm25_plus.fit(bm25_plus_scores)

            scaled_tf_idf_scores = (scaler_tf_idf.transform(tf_idf_scores)).flatten().tolist()  
            scaled_bm25_scores = (scaler_bm25.transform(bm25_scores)).flatten().tolist()
            scaled_bm25_plus_scores = (scaler_bm25_plus.transform(bm25_plus_scores)).flatten().tolist()
            
            # 각각의 min max scaling이 적용된 passage들을 한 곳에 합칩니다.합쳐진 전체 passage에서 top-k를 선정합니다.
            scaled_scores = scaled_tf_idf_scores + scaled_bm25_scores + scaled_bm25_plus_scores
            scaled_indices = self.tf_idf_indices[i] + self.bm25_indices[i] + self.bm25_plus_indices[i]
            
            # 합쳐진 전체 passage들을 min max scaling된 score를 기준으로 하여 내림차순으로 정렬합니다.
            score_index_pairs = list(zip(scaled_scores, scaled_indices))
            sorted_pairs = sorted(score_index_pairs, key=lambda pair: pair[0], reverse=True)
            selected_indices = []
            selected_scores = []

            # 정렬된(rerank된) passage들에 대해 중복되지 않게 top-k 개를 추출합니다.
            for score, index in sorted_pairs:
                if index not in selected_indices:
                    selected_indices.append(index)
                    selected_scores.append(score)

                    if len(selected_indices) >= self.topk:
                        break

            if len(selected_indices) < self.topk:
                raise ValueError("The number of selected documents is less than top-k.")
                
            all_selected_scores.append(selected_scores)
            all_selected_indices.append(selected_indices)
 
        # ensemble을 통해 rerank된 top-k개의 passage들을 반환합니다.
        if isinstance(self.datasets, str):
            for i in range(self.topk):
                print(f"Top-{i+1} passage with score {all_selected_scores[0][i]:4f}")
                print(self.contexts[all_selected_indices[0][i]])
            return (all_selected_scores, all_selected_indices)

        elif isinstance(self.datasets, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = all_selected_scores, all_selected_indices
            for idx, example in enumerate(tqdm(self.datasets, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                    tmp["scores"] = doc_scores[idx]
                if __name__ == "__main__":
                    tmp["scores"] = doc_scores[idx]
                    if tmp["original_context"] in tmp["context"]:
                        tmp["answer_index"] = [self.contexts[pid] for pid in doc_indices[idx]].index(tmp["original_context"])
                    else:
                        tmp["answer_index"] = -1
                
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


# Retriever Evaluation
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train_dataset", type=str, help="")
    parser.add_argument(
        "--model_name_or_path",
        default="klue/roberta-large",#"bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument("--retrieval_name", default="SparseRetrieval", type=str, help="")
    parser.add_argument("--top_k_retrieval", default=40, type=int, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    retriever.get_sparse_embedding()
    if args.retrieval_name == "SparseRetrieval":
        retriever = SparseRetrieval(tokenize_fn=tokenizer.tokenize, data_path=args.data_path, context_path=args.context_path)
        retriever.get_sparse_embedding()
    elif args.retrieval_name == "BM25":
        retriever = BM25(tokenize_fn=tokenizer.tokenize, data_path=args.data_path, context_path=args.context_path)
        retriever.get_sparse_embedding()
    elif args.retrieval_name == "BM25_Plus":
        retriever = BM25_Plus(tokenize_fn=tokenizer.tokenize, data_path=args.data_path, context_path=args.context_path)
        retriever.get_sparse_embedding()
    elif args.retrieval_name == "BM25_L":
        retriever = BM25_L(tokenize_fn=tokenizer.tokenize, data_path=args.data_path, context_path=args.context_path)
        retriever.get_sparse_embedding()
    elif args.retrieval_name == "Ensemble":
        print("Evaluate Retrieval Ensemble")
    else:
        raise ValueError("SparseRetrieval, BM25, BM25_Plus, BM25_L, Ensemble 중 하나를 정확히 입력해 주세요.")

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            if args.retrieval_name == "Ensemble":
                retriever = Retriever_Ensemble(datasets=full_ds, tokenize_fn=tokenizer.tokenize, data_path=args.data_path, context_path=args.context_path, topk=args.top_k_retrieval)
                df = retriever.ensemble_and_rerank()
            else:
                df = retriever.retrieve(full_ds, topk=args.top_k_retrieval)

            df["correct"] = [original_context in context for original_context, context in zip(df["original_context"], df["context"])]
            
            answer_scores = [] 
            scaler = MinMaxScaler()

            for scores, answer_index in zip(df["scores"], df["answer_index"]):
                if int(answer_index) >= 0:
                    scores_array = np.array(scores).reshape(-1, 1)
                    normalized_scores = scaler.fit_transform(scores_array).flatten()
                    answer_scores.append(normalized_scores[int(answer_index)])
            print(
                "retrival evaluation\n",
                f"retrieval({args.retrieval_name})가 retrive한 top-k({args.top_k_retrieval})개의 passage들 중에 정답이 포함되어 있는 경우 / 전체 query 수 : ",
                df["correct"].sum() / len(df),
            )
            '''참고자료 : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-09/issues/20'''
            print(
                " 정답 score들의 평균 : ",
                sum(answer_scores) / len(answer_scores),
                #answer_scores.sum() / len([score for score in answer_scores if score != 0]),
            )

        with timer("single query by exhaustive search"):
            if args.retrieval_name == "Ensemble": 
                retriever = Retriever_Ensemble(datasets=query, tokenize_fn=tokenizer.tokenize, data_path=args.data_path, context_path=args.context_path, topk=args.top_k_retrieval)
                scores, indices = retriever.ensemble_and_rerank()
            else:
                scores, indices = retriever.retrieve(query)