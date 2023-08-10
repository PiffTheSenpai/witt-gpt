from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

path = "EleutherAI/gpt-neox-20b"

model = GPTNeoXForCausalLM.from_pretrained(path)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(path)

prompt = open("./prompt.txt")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=250,
    line_by_line=True,
    to_gpu=True
)

a = open("./outputs.txt", "a")
gen_text = tokenizer.batch_decode(gen_tokens)[0]

