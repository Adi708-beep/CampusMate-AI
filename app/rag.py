from langchain_community.vectorstores import FAISS

from app.embeddings import get_embeddings
from app.llm import ask_campusmate


class CollegeRAG:

    def __init__(self):

        # Load embedding model
        embeddings = get_embeddings()

        # Load FAISS vector database
        self.vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

    def ask(self, question):

        # Retrieve relevant chunks
        documents = self.vectorstore.similarity_search(
            question,
            k=4
        )

        # Combine retrieved chunks into context
        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        # Generate answer
        answer = ask_campusmate(
            question,
            context
        )

        # Collect sources
        sources = []

        for doc in documents:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            if source not in sources:
                sources.append(source)

        return answer, sources