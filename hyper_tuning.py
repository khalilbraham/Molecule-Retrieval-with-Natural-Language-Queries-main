from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from info_nce import InfoNCE
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import wandb
import os

##os.environ['WANDB_API_KEY'] = 'WANDB_API_KEY'
#wandb.login()

def objective(trial):
    nb_epochs = trial.suggest_int("nb_epochs", 2, 10)
    batch_size = trial.suggest_int("batch_size", 32, 64)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    nhid = trial.suggest_int("nhid", 100, 1500)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    training_results = train_loop({
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "nhid": nhid,
        "dropout": dropout
    })

    val_loss = training_results['val_loss']
    return -val_loss

def train_loop(params):
    # Define CrossEntropyLoss and InfoNCE loss
    CE = torch.nn.CrossEntropyLoss()
    INCE = InfoNCE() 

    # Function for contrastive loss combining CE and INCE
    def contrastive_loss(v1, v2, beta=0.1):
        logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
        labels = torch.arange(logits.shape[0], device=v1.device)
        return beta * (CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)) + (1 - beta) * (INCE(v1, v2) + INCE(v2, v1))

    # Define the model_name and load pre-trained tokenizer
    # model_name = 'distilbert-base-uncased'
    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load node embeddings from file
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

    # Load validation and training datasets
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    nb_epochs = params['nb_epochs']
    batch_size = params['batch_size']
    graph_learning_rate = params['learning_rate']
    nhid = params['nhid']
    text_learning_rate = params['learning_rate']/10

    # Create DataLoader for validation and training datasets
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Model(model_name=model_name, num_node_features=300, nout=1024, nhid=nhid, graph_hidden_channels=1024, nheads=10) 
    model.to(device)

    # Define optimizer with different learning rates for graph and text encoders
    optimizer = optim.AdamW([{'params': model.graph_encoder.parameters()},
                            {'params': model.text_encoder.parameters(), 'lr': text_learning_rate}], lr=graph_learning_rate,
                            betas=(0.9, 0.999),
                            weight_decay=0.01)

    # Learning rate scheduler to adjust learning rate during training
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-7)

    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    best_validation_loss = 1000000

    # Training loop
    for i in range(nb_epochs):
        print('-----EPOCH{}-----'.format(i+1))
        model.train()
        for batch in train_loader:
            # Extract input data
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            # Forward pass
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            # Calculate contrastive loss
            current_loss = contrastive_loss(x_graph, x_text)
            optimizer.zero_grad()
            # Backward pass and optimization
            current_loss.backward()
            optimizer.step()
            loss += current_loss.item()

            count_iter += 1
            # Print training information
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/printEvery), flush=True)
                losses.append(loss)
                loss = 0
        
        # Validation
        model.eval()
        val_loss = 0
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            # Calculate contrastive loss for validation
            current_loss = contrastive_loss(x_graph, x_text)
            val_loss += current_loss.item()
        
        # Adjust learning rate based on validation loss
        lr_scheduler.step(val_loss)

        # Update best validation loss and save checkpoint if improved
        best_validation_loss = min(best_validation_loss, val_loss)
        print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) ,flush=True)
        if best_validation_loss==val_loss:
            print('validation loss improved saving checkpoint...')
            save_path = os.path.join('./models/', 'model'+str(i)+'.pt')
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))

study = optuna.create_study(direction="minimize")
study.optimize(objective)

# Initialize WandB with project name and run name
wandb.init(project="ALTEGRAD", name="optuna-tuning")

# Track hyperparameters and metrics
for key, value in study.best_params.items():
    wandb.log(key, value)

wandb.log("val_loss", study.best_value)
