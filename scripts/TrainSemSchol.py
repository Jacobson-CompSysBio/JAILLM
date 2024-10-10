# import necessary packages
import sys, os
import torch 
import numpy as np
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
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

# custom imports
from utils.GetLowestGPU import GetLowestGPU
device = GetLowestGPU()

## INIT MODEL
# options
model_path = "meta-llama/Meta-Llama-3-8B"
dataset_path = "allenai/peS2o"

accelerator = Accelerator()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

# load tokenizer and model
pipeline = pipeline('text-generation', 
                    model=model_path,
                    model_kwargs={'torch_dtype': torch.bfloat16},
                    device_map = accelerator.device
                    )

pipeline.model = get_peft_model(pipeline.model, peft_config)
pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
pipeline.model.generation_config.pad_token_id = pipeline.tokenizer.eos_token_id

pipeline.model.print_trainable_parameters()

# load dataset
raw_dataset = load_dataset(dataset_path, "v2", streaming=True, trust_remote_code=True)

# add special tokens to tokenizer
pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))

def main():

    # define functions
    def preprocess_data(examples):
        tokenized_data = pipeline.tokenizer(text=examples['text'],
                                            padding='max_length', 
                                            truncation=True, 
                                            max_length=100)
        return tokenized_data

    tokenized_dataset = raw_dataset.map(preprocess_data,
                                        batched=True,
                                        remove_columns=raw_dataset['train'].column_names,)
    tokenized_dataset.with_format("torch")

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
    
    # options
    num_batches = 10_000
    num_epochs = 5
    best_val_loss = np.inf
    checkpoint_path = '../checkpoints/checkpoint_{0}.pt'
    log_path = '../logs/log.csv'

    # init optimizer
    optimizer = AdamW(pipeline.model.parameters(), lr=1e-5)

    # init scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_epochs * num_batches,
    )

    pipeline.model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline.model, optimizer, train_dataloader, lr_scheduler)

    with open(log_path, 'w') as f: 
        f.write(f'epoch,iter_num,train_loss,val_loss\n')
    
    # run a test prediction
    text = ["Systems biology is the study of"]

    terminators = [
        pipeline.tokenizer.eos_token_id
    ]

    # loop
    for epoch in range(num_epochs):

        clear_output(wait=True)

        print("=====================")
        print(f"Epoch {epoch + 1}")
        print("=====================")

        # initialize train loss, val loss
        running_train_loss = 0.0
        running_val_loss = 0.0

        # loop through train data
        print("Training...")
        i = 0
        with tqdm(total=num_batches) as pbar:
            for train_batch, val_batch in zip(train_dataloader, val_dataloader):
                
                ## training
                # set model to train mode
                pipeline.model.train()

                # grab batch and map to device
                train_batch = {k: v.to(accelerator.device) for k, v in train_batch.items()}

                # forward pass
                outputs = pipeline.model(train_batch['input_ids'], 
                                        labels=train_batch['input_ids'],
                                        attention_mask=train_batch['attention_mask'])
                train_loss = outputs.loss

                running_train_loss += train_loss.item()

                # backward pass
                # train_loss.backward()
                accelerator.backward(train_loss)

                # clip gradients
                torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 1.0)

                # update optimizer, scheduler
                optimizer.step()
                lr_scheduler.step()

                # zero gradients
                optimizer.zero_grad()
                
                ## validation
                # set model to eval mode
                pipeline.model.eval()
                # loop through val data
                val_batch = {k: v.to(accelerator.device) for k, v in val_batch.items()}
                with torch.no_grad():
                    outputs = pipeline.model(val_batch['input_ids'], 
                                            labels=val_batch['input_ids'],
                                            attention_mask=val_batch['attention_mask'])
                    val_loss = outputs.loss
                    running_val_loss += val_loss.item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
                print(f"Train Batch Loss: {train_loss:.4f} | Val Batch Loss: {val_loss:.4f} | Best Val. Loss: {best_val_loss:.4f}\r", end="")

                i += 1
                pbar.update(1)
                if i % 1000 == 0:

                    # print example output
                    print(f"Batch {i} of {num_batches}; Printing Example Response...")
                    print(pipeline(text,
                                max_new_tokens=256,
                                eos_token_id=terminators,
                                no_repeat_ngram_size=3,       
                                do_sample=True, 
                                top_k=100, 
                                top_p=0.9,
                                temperature=0.6)[0][0]['generated_text'])

                # write to log
                with open(log_path, 'a') as f: 
                    f.write(f'{epoch},{i},{train_loss},{val_loss}\n')
                
                if i == num_batches:
                    print(f"Reached {num_batches} batches; breaking...")
                    break
        
        train_loss = running_train_loss / num_batches
        val_loss = running_val_loss / num_batches
        train_loss = running_train_loss / num_batches
        print(f"Avg. Train Loss: {train_loss:.4f}, Avg. Val Loss: {val_loss:.4f}")

    # save model checkpoint
    checkpoint = {'model': pipeline.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'iter_num': i,
                'best_val_loss': best_val_loss,
                }
    torch.save(checkpoint, checkpoint_path.format(i))
    print("Training Complete!")

if __name__ == "__main__":
    main()