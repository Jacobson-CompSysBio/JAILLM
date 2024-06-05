# import necessary packages
import sys, os
import torch 
import numpy as np
import evaluate
from trl import SFTTrainer, setup_chat_format
from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorWithPadding,
                          get_scheduler)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from IPython.display import clear_output

sys.path.append('../')

# custom imports
from utils.GetLowestGPU import GetLowestGPU

device = GetLowestGPU()

## ---------------------------------------------
## Define Functions for Preprocessing and Model
## ---------------------------------------------

def format_chat(row):
    row_json_inp = [{"role": "user", "content": row["Patient"]}]
    row_json_out = [{"role": "assistant", "content": row["Doctor"]}]
    row["input"] = tokenizer.apply_chat_template(row_json_inp, tokenize=False)
    row["target"] = tokenizer.apply_chat_template(row_json_out, tokenize=False)
    return row

def preprocess_data(examples):
    inp = examples["input"]
    out = examples["target"]
    tokenized_data = tokenizer(text=inp, 
                               text_target=out,
                               padding='max_length', 
                               truncation=True, 
                               max_length=512)
    return tokenized_data

if __name__ == "__main__":
``
    ## ---------------------------------------------
    ## Instantiate the Model, Tokenizer, and Dataset
    ## ---------------------------------------------

    # options
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_path = "ruslanmv/ai-medical-chatbot" #test dataset

    # load tokenizer and model
    pipeline = pipeline('text-generation', 
                        model=model_path,
                        model_kwargs={'torch_dtype': torch.bfloat16},
                        device_map = 'auto'
                        )

    model, tokenizer = pipeline.model, pipeline.tokenizer
    model, tokenizer = setup_chat_format(model, tokenizer)

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # load dataset and train/test split
    raw_dataset = load_dataset(dataset_path, split = 'train[:1%]')
    raw_dataset = raw_dataset.train_test_split(test_size=0.1)

    ## ---------------
    ## Preprocess Data
    ## ---------------

    chat_dataset = raw_dataset.map(format_chat)
    tokenized_dataset = chat_dataset.map(preprocess_data, 
                                        batched=True,
                                        remove_columns=chat_dataset['train'].column_names)
    tokenized_dataset.with_format("torch")

    # check tokenized dataset output
    tokenized_dataset

    # instantiate data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## ------------------
    ## Create Dataloaders
    ## ------------------

    # options
    batch_size = 8
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                batch_size=batch_size, 
                                collate_fn=data_collator)

    val_dataloader = DataLoader(tokenized_dataset['test'],
                                batch_size=batch_size,
                                collate_fn=data_collator)

    ## -----
    ## Train
    ## -----

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # and scheduler
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # loop
    for epoch in range(num_epochs):
        clear_output(wait=True)
        
        print("=====================")
        print(f"Epoch {epoch + 1}")
        print("=====================")

        # set model to train mode
        model.train()

        # initialize train loss, val loss
        train_loss = 0.0
        val_loss = 0.0

        # loop through train data
        print("Training...")
        for batch in train_dataloader:

            # grab batch and map to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward pass
            outputs = model(**batch)
            loss = outputs.loss

            train_loss += loss.item()

            # backward pass
            loss.backward()

            # update optimizer
            optimizer.step()

            # update scheduler
            lr_scheduler.step()

            # zero gradients
            optimizer.zero_grad()

        train_loss = train_loss / (len(train_dataloader) / batch_size)

        # set to eval mode
        model.eval()
        print("Validating...")
        for batch in val_dataloader:

            # get batch
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward pass
            with torch.no_grad():
                outputs = model(**batch)

            # get loss
            loss = outputs.loss
            val_loss += loss.item()


        val_loss = val_loss / (len(val_dataloader) / batch_size)

        print(f"Avg. Train Loss: {train_loss}, Avg. Val Loss: {val_loss}")
        # print("Evaluation metrics:", metric.compute())
        print("")

    ## ----
    ## Test
    ## ----

    message = 'I have a headache. What should I do?'

    # test after training
    text = [{'role': 'system', 'content': 'You are a helpful medical chatbot'},
            {'role': 'user', 'content': message}]
    print(pipeline(text, max_length=100)[0]['generated_text'])
