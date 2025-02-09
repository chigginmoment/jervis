# The LLM I am currently using
from transformers.utils import cached_file

model_name = "jeffmeloy/Qwen2.5-7B-nerd-uncensored-v1.0"
model_path = cached_file(model_name, "config.json") 

print(f"Model is stored at: {model_path}")
