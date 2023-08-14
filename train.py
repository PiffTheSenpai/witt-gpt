from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments

path = "EleutherAI/gpt-neo-125m"

config = AutoConfig.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(path)

 = open("./finetuning-data/phil-inv/Final_z_7-1-bez_cislovani.txt", "r")

training_args = TrainingArguments(
    output_dir="my_training",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4, 
    fp16=True, 
    save_total_limit=3,
    logging_steps=1,
    max_steps=80,
    optim="paged_adamw_8bit", 
    lr_scheduler_type="cosine", 
    warmup_ratio=0.05, 
    report_to="tensorboard",
)

trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    
)

trainer.train()
# !!zajimava moznost, cte trenovaci data radek po radku: line_by_line=True
# stejne tak tohle to_gpu=true
