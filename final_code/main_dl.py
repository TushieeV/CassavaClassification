import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader
from dataloader import CassavaDataset
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from epoch_train_val import train, validate
import os
from models import RNet, ResNet18, ResNet34, ResNet34Pre, ResNet50, NetConv
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from torchsummary import summary
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='ResNet34', help='NetConv, ResNet18, ResNet34, ResNet34Pre or ResNet50')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=2, help='scheduler patience')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--ndata', type=int, default=-1, help='total number of data points used')
    parser.add_argument('--factor', type=float, default=0.2, help='learning rate decay factor')
    parser.add_argument('--mname', type=str, default='final_model', help='file name to save model')
    args = parser.parse_args()

    df = pd.read_csv('train.csv')
    df = df.sample(frac=1, random_state=42) # Randomly shuffle data
    df = df[:args.ndata] if args.ndata != -1 else df

    df_train, df_valid = model_selection.train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'].values)
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    image_path = "./train_images"

    train_image_paths = np.array([os.path.join(image_path, x) for x in df_train['image_id'].values])
    valid_image_paths = np.array([os.path.join(image_path, x) for x in df_valid['image_id'].values])

    train_targets = df_train['label'].values
    valid_targets = df_valid['label'].values

    batch_size = 16

    cassava_train = None
    train_loader = None

    cassava_train = CassavaDataset(train_image_paths, train_targets, True) 
    train_loader = DataLoader(cassava_train, batch_size=batch_size, shuffle=False, num_workers=2)

    cassava_test = CassavaDataset(valid_image_paths, valid_targets, False) 
    test_loader = DataLoader(cassava_test, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = None
    logsoftmax = False

    if args.net == 'NetConv':
        model = NetConv(5)
        criterion = nn.NLLLoss() # For NetConv as it's outputs go through Log Softmax
        logsoftmax = True
    elif args.net == 'ResNet18':
        model = ResNet18(5)
        criterion = nn.CrossEntropyLoss() # For raw logits as ResNet outputs raw logits
    elif args.net == 'ResNet34':
        model = ResNet34(5)
        criterion = nn.CrossEntropyLoss()
    elif args.net == 'ResNet34Pre':
        model = ResNet34Pre(5)
        criterion = nn.CrossEntropyLoss()
    elif args.net == 'ResNet50':
        model = ResNet50(5)
        criterion = nn.CrossEntropyLoss() # For raw logits as ResNet outputs raw logits

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # Use GPU if available

    n_epochs = args.epochs
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    scheduler =  ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True, eps=1e-6)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_accuracy = 0
    best_epoch = 0

    # Main training process
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')

        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer, batch_size, logsoftmax=logsoftmax)
        val_loss, val_acc = validate(model, test_loader, device, criterion, batch_size, logsoftmax=logsoftmax)
        
        scheduler.step(val_loss.avg) # update scheduler with metric
        
        # Store training and test losses and accuracies
        train_losses.append(train_loss.avg)
        train_accs.append(train_acc.avg)
        val_losses.append(val_loss.avg)
        val_accs.append(val_acc.avg)

        if val_acc.avg > best_accuracy:
            best_accuracy = val_acc.avg
            best_epoch = epoch

    # Plot line plots for training and validation loss, accuracy

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['train', 'val'])
    plt.title('Model loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.show()

    plt.clf()

    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.legend(['train', 'val'])
    plt.title('Model accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy (%)', fontsize=15)
    plt.show()

    print(f'Best validation accuracy was: {best_accuracy}%')
    print(f'Best epoch was: {best_epoch}')

    # Save the model
    torch.save(model.state_dict(), args.mname)

if __name__ == '__main__':
    main()