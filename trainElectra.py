seed=16

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader
# conda install -c huggingface -c conda-forge datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy



eval_split = 'train' # train for do_train, validation for do_eval

# Adversarial squad addsent
    #dataset_id = tuple("squad_adversarial:AddSent".split(':'))
    #dataset = datasets.load_dataset(*dataset_id)
dataset = datasets.load_dataset('squad')#, split=eval_split) #split does separating into train and validation.
    #metric = datasets.load_metric(dataset)
model_class = AutoModelForQuestionAnswering
    #model="google/electra-small-discriminator"
task_kwargs={}
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator", use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained("google/electra-small-discriminator", **task_kwargs)
    # AutoModelForClassification needs num_labels= argument, but QA does not seem to require any.
    #print('tokenizer', tokenizer)

prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
train_dataset = None
eval_dataset = None
train_dataset_featurized = None
eval_dataset_featurized = None

NUM_PREPROCESSING_WORKERS = 8
train_dataset = dataset['train']
    
    # for stacking, training sets should be split into train and validation themselves.
max_train_samples= 10000
#if max_train_samples:
        ### use only 10,000 samples for training.
train_dataset = train_dataset.shuffle(seed=seed)#.select(range(max_train_samples))
        # eval_dataset will be those not selected in train_dataset.
        # use shuffle https://huggingface.co/docs/datasets/v2.15.0/process#shuffle
eval_dataset = train_dataset.shuffle(seed=seed).select(range(max_train_samples, len(dataset['train'])))
    ### small_
train_dataset_featurized = train_dataset.map( #https://huggingface.co/docs/datasets/about_map_batch
            prepare_train_dataset, #lambda exs: prepare_train_dataset_qa(exs, tokenizer)
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names 
            # make new columns using prepare_train_dataset_qa and remove the original columns.
        )
small_train_dataset_featurized = train_dataset_featurized.select(range(max_train_samples))
# since eval dataset here is actually part of the train dataset, so, it needs not apply prepare_eval_dataset_qa
eval_dataset_featurized = train_dataset_featurized.select(range(max_train_samples, len(train_dataset)))

#eval_dataset_featurized = train_dataset.select(range(max_train_samples)).select(range(max_train_samples, len(train_dataset))).map( #https://huggingface.co/docs/datasets/about_map_batch
#            prepare_train_dataset, #lambda exs: prepare_eval_dataset_qa(exs, tokenizer)
#            batched=True,
#            num_proc=NUM_PREPROCESSING_WORKERS,
#            remove_columns=train_dataset.column_names 
            # make new columns using prepare_train_dataset_qa and remove the original columns.
#        )

### set it to torch
small_train_dataset_featurized.set_format("torch")

#do_eval=True
#if do_eval:
        #eval_dataset = dataset[eval_split] # eval_dataset has been created above.
        #if args.max_eval_samples:
        #    eval_dataset = eval_dataset.select(range(args.max_eval_samples))
#    eval_dataset_featurized = eval_dataset.map(
#            prepare_eval_dataset,
#            batched=True,
#            num_proc=NUM_PREPROCESSING_WORKERS,
#            remove_columns=eval_dataset.column_names
#            )
        ### set it to torch
eval_dataset_featurized.set_format("torch")

trainer_class = Trainer
eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
compute_metrics = None

trainer_class = QuestionAnsweringTrainer

# eval_kwargs['eval_examples'] will be used in compute_metrics_and_store_predictions 
# to get the predictions from eval_dataset because it is not preprocessed.
eval_kwargs['eval_examples'] = eval_dataset
    # even if the task is qa, it does not always mean that the dataset is squad.
metric = datasets.load_metric('squad')
#import evaluate
#metric = evaluate.load("accuracy", "f1")
    # FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate

compute_metrics = lambda eval_preds: metric.compute(predictions=eval_preds.predictions, references=eval_preds.label_ids)
    
#eval_predictions = None
    
#def compute_metrics_and_store_predictions(eval_preds):
#        nonlocal eval_predictions
#        eval_predictions = eval_preds
#        return compute_metrics(eval_preds)
    
    ### DataLoader
    # https://huggingface.co/transformers/main_classes/trainer.html#customized-training
   
train_dataloader = DataLoader(small_train_dataset_featurized, shuffle=True, batch_size=64)
eval_dataloader = DataLoader(eval_dataset_featurized, batch_size=64)

### optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

### scheduler
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                    )

### send the model to cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device) #model = AutoModelForQuestionAnswering.from_pretrained("google/electra-small-discriminator", **task_kwargs)

### training
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):

    break

# ChatGPT says, loss can be calculated by the following.
#from torch.nn import CrossEntropyLoss
# Assuming 'start_positions' and 'end_positions' are the ground truth labels
# and 'outputs' is the result from your QA model
#start_logits, end_logits = outputs.start_logits, outputs.end_logits
#start_loss = CrossEntropyLoss()(start_logits, start_positions)
#end_loss = CrossEntropyLoss()(end_logits, end_positions)
#total_loss = (start_loss + end_loss) / 2
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


    #trainer.save_model()
###save model

#os.makedirs('./stacking_train', exist_ok=True)
#PATH=f"./stacking_train/electra-{0}.pt" # should be a file path, not a directory.
#torch.save(model.state_dict(), PATH)
#model.save(output_dir="./train_stacking/")
    
del model

#### load state_dict

n_models = 5
def load_all_state_dicts(n_models):
    all_state_dicts = list()
    ## you will have list them individually.
    for i in range(n_models):
       # define filename for this ensemble
       filename = f"./stacking_train/electra-{i}.pt"
       # load model from file
       state_dict = torch.load(filename)
       # add to list of members
       all_state_dicts.append(state_dict)
       print('>loaded %s' % filename)
    return all_state_dicts

# load all models
n_members = 5
members = load_all_state_dicts(n_members)
print('Loaded %d models in members' % len(members))

for batch in train_dataloader:
#    for i in range(64):
    #print(batch[i]) this yields a keyerror
    print(batch['start_positions'])
    break

# evaluate standalone models on test (residual of the train) dataset
#from helpers import postprocess_qa_predictions
#import evaluate
#metric = evaluate.load("accuracy", "f1")
model = AutoModelForQuestionAnswering.from_pretrained("google/electra-small-discriminator")
# use cuda if available
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print('ok')
# let's skip evaluation for now.
#model.load_state_dict(members[0])
#model.eval()


#### create inputs, (which are the outputs of base models), for the meta-learner

from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
num_training_steps = len(eval_dataloader)
progress_bar = tqdm(range(num_training_steps))

stackX = None
stackY = None

# load the fifth state_dict
model.load_state_dict(members[4])
model.eval()  # Set the model to evaluation mode

for i, batch in enumerate(eval_dataloader):
	
	batch = {k: v.to(device) for k, v in batch.items()}
	with torch.no_grad():  # Ensure no gradient is computed
		outputs = model(**batch)#.detach()  # Detach from computation graph
	start_logits, end_logits = outputs.start_logits, outputs.end_logits
	#print(f'start_logits.shape: {start_logits.shape}') #torch.Size([64, 512])
	
### compute loss
	#start_loss = CrossEntropyLoss()(start_logits, batch['start_positions'])
	#end_loss = CrossEntropyLoss()(end_logits, batch['end_positions'])
	#total_loss = (start_loss + end_loss) / 2
	
	#prediction_start = torch.argmax(start_logits, dim=-1)
	#print(f'prediction_start.shape: {prediction_start.shape}') # torch.Size([64])
	#prediction_end = torch.argmax(end_logits, dim=-1) # is 1 needed? #+ 1
	#print(f'prediction_end: {prediction_end.shape}') # torch.Size([64])
	#if prediction_start.is_cuda:
	#	prediction_start = prediction_start.cpu()  # Move to CPU if on GPU
	#if prediction_end.is_cuda:
	#	prediction_end = prediction_end.cpu()
	#prediction_start_np = prediction_start.numpy()
	start_logits_np = start_logits.cpu().numpy()
	#prediction_end_np = prediction_end.numpy()
	end_logits_np = end_logits.cpu().numpy()

	# stack predictions into [rows, members, probabilities]
	if stackX is None:
		stackX = start_logits_np
		print(f'inital stackX shape: {stackX.shape}') #(64,512)
	else:
		stackX = np.vstack((stackX, start_logits_np))
		# dstack should be done at the inter-model levels, not within the same model.
		print(f'stackX shape after vstack: {stackX.shape}') 
		# must be 64*1212+31,512: 64 batch size* i iterations, 512: sequence length
	if stackY is None:
		stackY = end_logits_np
	else:
		stackY = np.vstack((stackY, end_logits_np))

	#print(f'After batch {i}, stackX shape: {stackX.shape}')
	progress_bar.update(1)
	#if i==3:
	#	break
	# flatten predictions to [rows, members x probabilities]
print(f'stackX.shape: {stackX.shape}') # must be 64*i,512: 64 batch size* i iterations, 512: sequence length

## reshape predictions to [rows x members, probabilities]
#stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
#stackY = stackY.reshape((stackY.shape[0], stackY.shape[1]*stackY.shape[2]))
#print(f'stackX.shape: {stackX.shape}') # must be 1, 64*i: 1 member, 64 batch size* i iterations

start_logit_dstack = np.dstack((start_logit_dstack, stackX))
print(f'start_logit_dstack shape after dstack: {start_logit_dstack.shape}') 
end_logit_dstack = np.dstack((end_logit_dstack, stackY))
print(f'end_logit_dstack shape after dstack: {end_logit_dstack.shape}')


### save with Pickle

import pickle

# Save a variable
#my_data = {start_logit_dstack, end_logit_dstack}
with open('start_logit_dstack.pkl', 'wb') as f:
    pickle.dump(start_logit_dstack, f)
with open('end_logit_dstack.pkl', 'wb') as f:
    pickle.dump(end_logit_dstack, f)


#### 1. use logistic regression as meta learner

# fit a model based on the outputs from the ensemble members
from sklearn.linear_model import LogisticRegression
model_start = LogisticRegression()
model_start.fit(start_logit_dstack, eval_dataset_featurized['start_positions'])




torch.cuda.empty_cache()
#del trainer
