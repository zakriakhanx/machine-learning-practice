from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("Ahren09/empathetic_dialogues")

def preprocess_function(examples):
    # We want to pair the Situation (prompt) with the first dialogue line (utterance)
    # OR better: pair the user's line with the empathetic reply.
    
    inputs = []
    for prompt, utterance in zip(examples['prompt'], examples['utterance']):
        # We format it so the model learns to respond TO the situation
        inputs.append(f"User: {prompt} Bot: {utterance}")
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# RE-TRAIN with this mapping!
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# IMPORTANT: Remove the old columns so the Trainer only sends 
# input_ids, attention_mask, and labels to the model
tokenized_dataset = tokenized_dataset.remove_columns(dataset["train"].column_names)

print(tokenized_dataset["train"][0].keys())
# You SHOULD see: dict_keys(['input_ids', 'attention_mask', 'labels'])



model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./empathetic-chatbot",
    eval_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    num_train_epochs=3,
    fp16=True,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

print("Starting training...")
trainer.train()

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("Model saved successfully!")