from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.embeddings import embeddings


DOCUMENT_FOLDER = Path("data/documents")


def load_documents():

    documents = []

    for file in DOCUMENT_FOLDER.glob("*.pdf"):

        print(f"Loading: {file.name}")

        loader = PyPDFLoader(str(file))

        docs = loader.load()

        documents.extend(docs)

    return documents


def main():

    print("Loading college documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} pages."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating embeddings...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        "vectorstore"
    )

    print("Vector database created successfully!")

if __name__ == "__main__":
    main()