from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

path = "EleutherAI/gpt-neox-20b"

config = AutoConfig.from_pretrained(path)
model = GPTNeoXForCausalLM.from_pretrained(path)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(path)

prompt = open("./prompt.txt")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_new_tokens=250,
    #line_by_line=True,
    #to_gpu=True,
)

#a = open("./outputs.txt", "a")
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
