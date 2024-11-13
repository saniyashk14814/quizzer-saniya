# test_torch.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from transformers import pipeline
nlp = pipeline("text-generation", model="gpt2", device=-1)
print("Transformers pipeline created successfully")