import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --------------------------------
# Device Setup
# --------------------------------
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CUDA not available. Using CPU.")


# --------------------------------
# Load Tokenizer & Model
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
)

if device == "cuda":
    model = model.to("cuda")


# --------------------------------
# Pipeline Initialization
# --------------------------------
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False,
)


def ask_campusmate(question, context):
    """
    Generate an accurate RAG answer using local Qwen2.5-0.5B-Instruct.
    Uses ChatML template to guarantee strict adherence to context and rules.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are CampusMate AI, a college assistant.\n\n"
                "Answer the student's question using ONLY the information provided in the college knowledge base.\n\n"
                "Rules:\n"
                "1. Do not make up information.\n"
                "2. Do not use outside knowledge.\n"
                "3. If the answer is not available in the context, say:\n"
                '"I could not find this information in the college knowledge base."\n'
                "4. Give a detailed but easy-to-understand answer.\n"
                "5. Stay focused on the student's question.\n"
                "6. If the question asks for names, dates, departments, rules, or other specific information, "
                "provide the information clearly in a list when appropriate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"College Knowledge Base:\n\n{context}\n\n"
                f"Student Question:\n{question}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = pipe(
        prompt,
        max_new_tokens=300,
        return_full_text=False,
    )

    return outputs[0]["generated_text"].strip()