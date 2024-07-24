import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

from type import QA

load_dotenv('../.env')

class QADatabase:

    def __init__(self, collection_name='question') -> None:
        self.get_collection(collection_name)

    def get_collection(self, name: str) -> bool:
        chroma_client = chromadb.PersistentClient(path='../db')

        openai_ef = (
            embedding_functions
            .OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name='text-embedding-3-small',
            )
        )

        self.collection = (
            chroma_client
            .get_collection(
                name=name,
                embedding_function=openai_ef,
            )
        )

    def get_similar_example(self, query: list[str], n_results=3) -> list[QA]:
        results = (
            self.collection
            .query(
                query_texts=query,
                n_results=n_results,
            )
        )

        top_n_example = list()
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        for q, a in zip(documents, metadatas):
            top_n_example.append(QA(q, a['answer']))

        return top_n_example
