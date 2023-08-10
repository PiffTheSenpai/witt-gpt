from transformers import AutoTokenizer, AutoModelForCausalLM

path = "EleutherAI/gpt-neox-20b"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)

open ("./finetuning-data/phil-inv/Final_z_7-1-bez_cislovani.txt")

# !!zajimava moznost, cte trenovaci data radek po radku: line_by_line=True
# stejne tak tohle to_gpu=true
