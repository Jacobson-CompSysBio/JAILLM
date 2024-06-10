# import necessary packages
import sys, os
import torch 
import numpy as np
import evaluate
from trl import SFTTrainer, setup_chat_format
from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorForLanguageModeling,
                          DataCollatorWithPadding,
                          get_scheduler)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from IPython.display import clear_output

sys.path.append('../')

# SPECIFY DEVICE AS NEEDED....
device = ...

## DEFINE FUNCTIONS

# define functions
def preprocess_data(examples):

    tokenized_data = pipeline.tokenizer(text=examples['text'],
                               padding='max_length', 
                               truncation=True, 
                               max_length=64)
    
    labels = tokenized_data['input_ids'].copy()

    for i in range(len(labels)):
        if labels[i][-1] != pipeline.tokenizer.pad_token_id:
            labels[i] = labels[i][1:] + [pipeline.tokenizer.pad_token_id]
        else:
            labels[i] = labels[i][1:] + [-100]

    labels = [[-100 if x == pipeline.tokenizer.pad_token_id else x for x in y] for y in labels]
    tokenized_data['labels'] = labels
    
    return tokenized_data

if name == "__main__":

    ## INSTANTIATE MODEL & DATASET

    # options
    model_path = "meta-llama/Meta-Llama-3-8B"
    dataset_path = "allenai/peS2o"

    # load tokenizer and model
    pipeline = pipeline('text-generation', 
                        model=model_path,
                        model_kwargs={'torch_dtype': torch.bfloat16},
                        device_map = 'auto'
                        )

    # load dataset
    raw_dataset = load_dataset(dataset_path, "v2", streaming=True, trust_remote_code=True)

    # check format of data
    raw_dataset

    ## PREPROCESS DATA

    # add special tokens to tokenizer
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))

    tokenized_dataset = raw_dataset.map(preprocess_data,
                                        batched=True,
                                        remove_columns=raw_dataset['train'].column_names,)
    tokenized_dataset.with_format("torch")

    # check tokenized dataset output
    tokenized_dataset

    ## CREATE DATALOADERS

    # instantiate data collator
    data_collator = DataCollatorWithPadding(tokenizer=pipeline.tokenizer)

    train_dataloader = DataLoader(tokenized_dataset['train'],
                                batch_size=8, 
                                collate_fn=data_collator,
                                num_workers=20)

    val_dataloader = DataLoader(tokenized_dataset['validation'],
                                batch_size=8,
                                collate_fn=data_collator,
                                num_workers=2)

    ## TRAIN MODEL

    # run a test prediction
    messages = ["Network biology is"]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs)

    # options
    optimizer = AdamW(pipeline.model.parameters(), lr=1e-5)
    num_epochs = 3

    # loop
    for epoch in range(num_epochs):
        
        print("=====================")
        print(f"Epoch {epoch + 1}")
        print("=====================")

        # set model to train mode
        pipeline.model.train()

        # initialize train loss, val loss
        running_train_loss = 0.0
        running_val_loss = 0.0

        # loop through train data
        print("Training...")
        i = 0
        for batch in train_dataloader:

            # grab batch and map to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward pass
            outputs = pipeline.model(**batch)
            loss = outputs.loss
            print(f"batch loss: {loss:.4f}\r", end="")

            running_train_loss += loss.item()

            # backward pass
            loss.backward()

            # update optimizer
            optimizer.step()

            # zero gradients
            optimizer.zero_grad()

            i += 1
            if i % 1000 == 0:
                print(f"Processed {i} batches; Printing example response...")
                print(pipeline(messages, max_length=100, truncation=True))
            
        # set model to eval mode
        pipeline.model.eval()

        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = pipeline.model(**batch)
                loss = outputs.loss
                running_val_loss += loss.item()
            
        val_loss = running_val_loss / len(val_dataloader)

        print("Printing example response...")
        print(pipeline(messages, max_length=100, truncation=True))

        train_loss = running_train_loss / len(train_dataloader)
        print(f"Avg. Train Loss: {train_loss:.4f}, Avg. Val Loss: {val_loss:.4f}")
        # print("Evaluation metrics:", metric.compute())

    print("Training Complete!")
