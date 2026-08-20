from langchain_community.vectorstores import FAISS

from app.embeddings import get_embeddings
from app.llm import ask_campusmate


class CollegeRAG:

    def __init__(self):

        embeddings = get_embeddings()

        self.vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

    def ask(self, question):

        documents = self.vectorstore.similarity_search(
            question,
            k=4
        )

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        answer = ask_campusmate(
            question,
            context
        )

        sources = []

        for doc in documents:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            if source not in sources:
                sources.append(source)

        return answer, sources