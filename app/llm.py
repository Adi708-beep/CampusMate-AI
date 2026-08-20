from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# Create local text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=600,
    temperature=0.2,
    do_sample=True,
    return_full_text=False
)


# Convert to LangChain model
llm = HuggingFacePipeline(
    pipeline=pipe
)


def ask_campusmate(question, context):

    prompt = f"""
You are CampusMate AI, a college assistant.

Answer the student's question using ONLY the
information provided in the college knowledge base.

Rules:
1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:

"I could not find this information in the college knowledge base."

4. Give a detailed but easy-to-understand answer.
5. Stay relevant to the student's question.

College Knowledge Base:
{context}

Student Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response