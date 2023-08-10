from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/storage/brno2/home/tvrzj/EleutherAI/gpt-neox-20b")
model = AutoModel.from_pretrained("/storage/brno2/home/tvrzj/EleutherAI/gpt-neox-20b")

open ("./finetuning-data/phil-inv/Final_z_7-1-bez_cislovani.txt")

# !!zajimava moznost, cte trenovaci data radek po radku: line_by_line=True
# stejne tak tohle to_gpu=true
