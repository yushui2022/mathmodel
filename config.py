import os
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ========= 基础模型与向量库 =========
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 嵌入模型
embeddings = OllamaEmbeddings(
    model="bge-m3:latest",
    base_url=OLLAMA_BASE_URL,
)

# 重排序模型
rerank_embeddings = OllamaEmbeddings(
    model="qllama/bge-reranker-v2-m3:latest",
    base_url=OLLAMA_BASE_URL,
)

# 大模型
llm = ChatOllama(
    model="llama3:latest",
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    num_ctx=6144,
    num_predict=2048,
)

# 向量库
vectorstore = FAISS.load_local(
    "math_modeling_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# 初筛检索器（重排序后会再裁剪）
retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

__all__ = [
    "embeddings",
    "rerank_embeddings",
    "llm",
    "vectorstore",
    "retriever",
    "OLLAMA_BASE_URL",
]

