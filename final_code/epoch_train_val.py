import torch
from utils import AverageMeter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import time
import datetime
from models import NetConv
import numpy as np

# Model training function
def train(model, loader, device, loss_fn, optimizer, batch_size, logsoftmax=False):
    model.train()
    summary_loss = AverageMeter() 
    summary_acc = AverageMeter()
    losses = []
    start = time.time() 
  
    pos_accs = []
    neg_accs = []

    for batch in loader:

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)           
        loss = loss_fn(out, labels)   

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():

            if logsoftmax:
                # For log softmax output
                probs = torch.exp(out)
            else:
                # For raw logits
                probs = torch.softmax(out, dim=1)

            probs = probs.cpu().numpy()
            acc = accuracy_score(np.argmax(probs, axis=1), labels.cpu().numpy()) * 100

        summary_loss.update(loss.detach().item(), batch_size)
        summary_acc.update(acc, batch_size)
        
    train_time = str(datetime.timedelta(seconds=time.time() - start))
    print(f'Train loss: {round(summary_loss.avg, 5)} - Train accuracy: {round(summary_acc.avg, 2)}% - Time: {train_time}')
    return summary_loss, summary_acc

# Model validation function
def validate(model, loader, device, loss_fn, batch_size, logsoftmax=False):
    model.eval()
    summary_loss = AverageMeter()
    summary_acc = AverageMeter() 
    start = time.time() 
    
    for batch in loader:
        with torch.no_grad():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)                  
            loss = loss_fn(out, labels)    

            if logsoftmax:
                # For log softmax output
                probs = torch.exp(out)
            else:
                # For raw logits
                probs = torch.softmax(out, dim=1)
            
            probs = probs.cpu().numpy()
            acc = accuracy_score(np.argmax(probs, axis=1), labels.cpu().numpy()) * 100

            summary_loss.update(loss.detach().item(), batch_size)
            summary_acc.update(acc, batch_size)
    
    eval_time = str(datetime.timedelta(seconds=time.time() - start))
    print(f'Val loss: {round(summary_loss.avg, 5)} - Val accuracy: {round(summary_acc.avg, 2)}% - Time: {eval_time}')
    return summary_loss, summary_acc