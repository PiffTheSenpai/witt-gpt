from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/storage/brno2/home/tvrzj/EleutherAI/gpt-neox-20b")
model = AutoModel.from_pretrained("/storage/brno2/home/tvrzj/EleutherAI/gpt-neox-20b")
