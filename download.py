# download modelu a ulozeni do lokalu
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.save_pretrained("/storage/brno2/home/tvrzj/EleutherAI/gpt-neox-20b")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
model.save_pretrained("/storage/brno2/home/tvrzj/EleutherAI/gpt-neox-20b")

''' # nakonec ne, zjistil jsem ze je to jen jina metoda
# stazeni configu
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
'''