from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
MODEL_DIR = "/opt/app-root/src/tinyllama-dell-fast-final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,  # CPU
)

app = FastAPI(title="Dell TinyLLaMA API")

class RequestData(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@app.post("/generate")
def generate_text(data: RequestData):
    inputs = tokenizer(data.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=data.max_new_tokens,
        temperature=data.temperature,
        top_p=data.top_p,
        do_sample=data.do_sample
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": result}

@app.get("/health")
def health_check():
    return {"status": "ok"}
