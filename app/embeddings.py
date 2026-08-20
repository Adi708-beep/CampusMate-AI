from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()


def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embeddings


if __name__ == "__main__":

    embeddings = get_embeddings()

    result = embeddings.embed_query(
        "What is the academic structure?"
    )

    print(result)