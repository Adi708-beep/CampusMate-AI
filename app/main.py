import sys
from pathlib import Path


# --------------------------------
# Add project root to Python path
# --------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR)
    )


import streamlit as st

from app.rag import CollegeRAG


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(

    page_title="CampusMate AI",

    page_icon="🎓",

    layout="centered"
)


# --------------------------------
# Header
# --------------------------------

st.title("🎓 CampusMate AI")

st.write(
    "Your AI-powered college assistant"
)

st.caption(
    "Ask questions about academics, exams, "
    "library, scholarships and college rules."
)


# --------------------------------
# Load RAG
# --------------------------------

@st.cache_resource
def load_rag():

    return CollegeRAG()


try:

    rag = load_rag()

except Exception as e:

    st.error(
        "Unable to load the CampusMate "
        "knowledge base."
    )

    st.exception(e)

    st.info(
        "Make sure your FAISS vectorstore "
        "has been created."
    )

    st.stop()


# --------------------------------
# Chat History
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------
# Display Chat History
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.caption("📄 Sources:")

            for source in message["sources"]:

                st.caption(
                    f"• {source}"
                )


# --------------------------------
# User Question
# --------------------------------

question = st.chat_input(
    "Ask CampusMate something..."
)


if question:

    # Store question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Display question
    with st.chat_message("user"):

        st.markdown(question)


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching college knowledge..."
        ):

            try:

                answer, sources = rag.ask(
                    question
                )


                st.markdown(answer)


                # Display sources
                if sources:

                    st.caption(
                        "📄 Sources:"
                    )

                    for source in sources:

                        st.caption(
                            f"• {source}"
                        )


                # Store answer
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )