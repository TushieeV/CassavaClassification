import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from collections import Counter
import torch
from models import ResNet34, ResNet34Pre
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

# code largely taken from https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
def feature_map(model, img):
    
    model_weights = []
    conv_layers = []

    model_children = list(model.children())
    
    # counter to keep count of the conv layers
    counter = 0 
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    pop_mean = [0.4308398 , 0.49935585, 0.31198692]
    pop_std = [0.22837807, 0.2308237 , 0.21775971]

    # define the transforms
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pop_mean, std=pop_std)
    ])
    # apply the transforms
    img = transform(img)
    print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize 64 features from first, middle and end layers
    # (although there are more feature maps in the upper layers)
    #for num_layer in range(len(outputs)):
    for num_layer in [0, 17, 32]:
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            #plt.imshow(filter, cmap='gray')
            plt.imshow(filter)
            plt.axis("off")
        plt.show()

def label_histogram(df):
    print(Counter(df['label']))
    df.hist(column='label', bins=np.arange(6) - 0.5, edgecolor='black')
    plt.title('Histogram of labels', fontsize=20)
    plt.xticks([0, 1, 2, 3, 4])
    plt.grid(b=None)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.show()

def plot_class(df, lbl):

    diseases = {
        0: "Cassava Bacterial Blight (CBB)", 
        1: "Cassava Brown Streak Disease (CBSD)", 
        2: "Cassava Green Mottle (CGM)", 
        3: "Cassava Mosaic Disease (CMD)", 
        4: "Healthy"
    }

    images = df.loc[df['label'] == lbl]['image_id'].values

    plot_imgs = [mpimg.imread(f'./train_images/{images[np.random.randint(0, len(images))]}') for _ in range(4)]

    f, ax = plt.subplots(2,2)
    f.suptitle(f'{lbl} : {diseases[lbl]}', fontsize=20)
    ax[0,0].imshow(plot_imgs[0])
    ax[0,1].imshow(plot_imgs[1])
    ax[1,0].imshow(plot_imgs[2])
    ax[1,1].imshow(plot_imgs[3])

    plt.show()


df = pd.read_csv('train.csv')

# Plot examples of class 2
#plot_class(df, 2)

# Plot label distribution
#label_histogram(df)

# Plot final model feature maps of layers 0, 17 and 32
#model = ResNet34(5)
#model.load_state_dict(torch.load('final_model_85_acc', map_location='cpu'))
#class_0_img = Image.open('train_images/1012426959.jpg')
#feature_map(model, class_0_img)