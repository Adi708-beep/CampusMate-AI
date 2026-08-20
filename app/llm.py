import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from langchain_huggingface import HuggingFacePipeline




MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# --------------------------------
# Check device
# --------------------------------

if torch.cuda.is_available():

    device = "cuda"

    print("Using GPU:")
    print(torch.cuda.get_device_name(0))

else:

    device = "cpu"

    print("CUDA not available.")
    print("Using CPU.")


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------
# Load model
# --------------------------------

if device == "cuda":

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )

    model = model.to("cuda")

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )



model.generation_config.max_length = None


# --------------------------------
# Text generation pipeline
# --------------------------------

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    return_full_text=False
)



llm = HuggingFacePipeline(
    pipeline=pipe
)


def ask_campusmate(question, context):

    prompt = f"""
You are CampusMate AI, a college assistant.

Answer the student's question using ONLY
the information provided in the college
knowledge base.

Rules:

1. Do not make up information.

2. Do not use outside knowledge.

3. If the answer is not available in the
context, say:

"I could not find this information in the
college knowledge base."

4. Give a detailed but easy-to-understand answer.

5. Stay focused on the student's question.

6. If the question asks for names, dates,
departments, rules, or other specific information,
provide the information clearly in a list when
appropriate.

College Knowledge Base:

{context}

Student Question:

{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response