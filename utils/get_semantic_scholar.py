## import packages
import copy
import datasets
import itertools


def get_semantic_scholar(dataset_config, tokenizer, split: str):

    """
    ----------------------------------------------
    Returns a custom dataset for use with torchrun
    ----------------------------------------------

    Parameters:
        dataset_config (dict): A dictionary containing the dataset configuration options
        tokenizer (transformers.tokenizer): A tokenizer object
        split (str): The split of the dataset to use (train/val/test)

    Returns:
        dataset: A custom dataset object
    """

    dataset = datasets.load_dataset("allenai/peS2o", split=split)

    def preprocess_function(examples):
        tokenized_data = tokenizer(text=examples['text'],
                                padding='max_length', 
                                truncation=True, 
                                max_length=64)
        
        labels = tokenized_data['input_ids'].copy()

        for i in range(len(labels)):
            if labels[i][-1] != tokenizer.pad_token_id:
                labels[i] = labels[i][1:] + [tokenizer.pad_token_id]
            else:
                labels[i] = labels[i][1:] + [-100]

        labels = [[-100 if x == tokenizer.pad_token_id else x for x in y] for y in labels]
        tokenized_data['labels'] = labels
        return tokenized_data

    dataset = dataset.map(preprocess_function,
                          batched=True, 
                          remove_columns=dataset[split].column_names)
    
    return dataset