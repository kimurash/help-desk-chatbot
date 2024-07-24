import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import pandas as pd

load_dotenv('../.env')

chroma_client = chromadb.PersistentClient(path='../db')
chroma_client.heartbeat()

df_qa = pd.read_csv('../qa.csv')

openai_ef = (
    embedding_functions
    .OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name='text-embedding-3-small',
    )
)
collection = (
    chroma_client
    .get_or_create_collection(
        name='question',
        embedding_function=openai_ef,
        # cosine類似度を採用
        metadata={"hnsw:space": "cosine"},
    )
)
collection.add(
    documents=df_qa['Q'].tolist(),
    # 回答をmetadataとして保存
    metadatas=[{'answer': answer} for answer in df_qa['A'].tolist()],
    ids=[str(id) for id in range(len(df_qa))],
)
