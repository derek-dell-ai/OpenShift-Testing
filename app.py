from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import torch

MODEL_PATH = "/models/tinyllama/tinyllama-dell-fast-final"

app = FastAPI()

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.7

@app.post("/generate")
def generate_text(query: Query):
    inputs = tokenizer(query.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=query.max_new_tokens,
            temperature=query.temperature,
            do_sample=True
        )
    return {"text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
