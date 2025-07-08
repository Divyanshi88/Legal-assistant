import os
import shutil
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# 📂 File and folder paths
PDF_FILE = "data/Womenrights.pdf"
CHROMA_PATH = "chroma_store"
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


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"✅ Saved {len(chunks)} chunks to `{CHROMA_PATH}`.")


def generate_data_store():
    documents = load_documents()
    if not documents:
        print("❌ No documents found. Check the file path.")
        return
    chunks = split_text(documents)
    save_to_chroma(chunks)


def main():
    generate_data_store()


if __name__ == "__main__":
    main()
