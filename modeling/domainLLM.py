import json
import time
from datasets import Dataset
from pathlib import Path
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from util import tokenizer, save_model, tokenize_function, split_into_chunks

indir = Path("../../collections")

documents = []

for file in indir.iterdir():
    if file.suffix.lower() == ".json":
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data:
            for speech in obj["interviewee speech"]:  # obj["interviewee speech"] is an array of interviewee responses
                chunks = split_into_chunks(speech)
                for chunk in chunks:
                    documents.append({"text": tokenizer.decode(chunk)})

dataset = Dataset.from_list(documents).train_test_split(test_size=0.1)  # 90% train, 10% valid
    
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

model = RobertaForMaskedLM.from_pretrained("roberta-base")
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=None,
        logging_dir=None,
        num_train_epochs=10,
        weight_decay=1e-3,
        logging_steps=10,
        learning_rate=3e-5,
        save_strategy="no",
        log_level='info',
        disable_tqdm=False,
        eval_strategy="epoch"
    ),
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

start_time = time.time()
trainer.train()
end_time = time.time()
print(f"fine-tuning domain adaptation for 10 epochs took {end_time - start_time:.2f} seconds.")
save_model(model, f"roberta_domain_adapted")
