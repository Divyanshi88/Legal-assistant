import os
import shutil
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS

# Load environment variables with fallback for deployment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 📂 File and folder paths
PDF_FILE = "data/Womenrights.pdf"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from the PDF.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=[r"\n\d+\.\s", "\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split {len(documents)} pages into {len(chunks)} chunks.")

    processed_chunks = []
    for i, chunk in enumerate(chunks):
        match = re.search(r"(Section|section)\s*(\d+[A-Za-z]*)", chunk.page_content)
        section_id = match.group(2) if match else None

        chunk.metadata.update({
            "chunk_id": i,
            "chunk_size": len(chunk.page_content),
            "section": section_id,
            "document_type": "legal_document",
            "subject": "domestic_violence_act"
        })
        processed_chunks.append(chunk)

    if processed_chunks:
        print("📄 Sample chunk:")
        print(processed_chunks[0].page_content[:400])
        print("Metadata:", processed_chunks[0].metadata)

    return processed_chunks


def save_to_faiss(chunks: list[Document]):
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = FAISS.from_documents(chunks, embedding)
    db.save_local(FAISS_INDEX_PATH)
    print(f"✅ Saved {len(chunks)} chunks to FAISS index at `{FAISS_INDEX_PATH}`.")


def generate_data_store():
    documents = load_documents()
    if not documents:
        print("❌ No documents found. Check the file path.")
        return
    chunks = split_text(documents)
    save_to_faiss(chunks)


def main():
    generate_data_store()


if __name__ == "__main__":
    main()
