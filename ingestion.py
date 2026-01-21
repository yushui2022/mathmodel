import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    print("Ingesting...")
    # 读取项目内文本，避免硬编码绝对路径（同时指定 encoding="utf-8" 解决解码错误）
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediumblog1.txt")
    loader = TextLoader(input_path, encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # 减小chunk大小以适应Ollama上下文长度限制
        chunk_overlap=50,  # 保持合理的重叠
        separators=["\n\n", "\n", "。", "，", " ", ""]  # 按段落、句子、标点符号分层拆分
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # 初始化本地Ollama嵌入模型
    embeddings = OllamaEmbeddings(
        model="bge-m3:latest",
        base_url="http://localhost:11434"
    )

    print("ingesting to local FAISS...")
    # 检查分块结果
    for i, chunk in enumerate(texts[:3]):  # 只检查前3个chunk
        print(f"Chunk {i+1} size: {len(chunk.page_content)} characters")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index_mediumblog_new")  # 使用新文件夹名

    print("finish! 索引文件已生成：faiss_index_mediumblog_new")