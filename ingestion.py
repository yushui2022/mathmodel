import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

if __name__ == "__main__":
    print("Ingesting...")
    # ========== 核心修复：添加 encoding="utf-8" 参数 ==========
    loader = TextLoader(
        "C:/Users/xiaoy/Desktop/langchain-course-project-rag-gist/mediumblog1.txt",
        encoding="utf-8"  # 指定文件编码为UTF-8，解决解码错误
    )
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,  # 增加重叠，避免拆分截断语义
        separator="\n\n"  # 按空行拆分，更符合文本语义
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # 初始化本地Ollama嵌入模型
    embeddings = OllamaEmbeddings(
        model="bge-m3:latest",
        base_url="http://localhost:11434"
    )

    print("ingesting to local FAISS...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index_mediumblog")

    print("finish! 索引文件已生成：faiss_index_mediumblog")