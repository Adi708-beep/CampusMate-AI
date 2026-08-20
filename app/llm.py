import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()





llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    temperature=0.2,
    max_new_tokens=300,
)

model = ChatHuggingFace(llm=llm)




def ask_campusmate(question, context):

    prompt = f"""
You are CampusMate AI, a college assistant.

Your job is to answer the student's question using
ONLY the information provided in the college knowledge base.

Rules:
1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not available in the context,
   say exactly:

"I could not find this information in the college knowledge base."

4. Keep the answer short and easy to understand.
5. Answer directly without unnecessary explanation.

College Knowledge Base:
{context}

Student Question:
{question}

Answer:
"""

    response = model.invoke(prompt)

    return response.content