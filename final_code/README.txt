Tushar Virk (z5208092)

This README contains descriptions of all relevant files used for the COMP9417 project.

NOTE: The dataset which consists of the following files: 
	- test_images
	- test_tfrecords
	- train_images
	- train_tfrecords
	- train.csv
	- sample_submission.csv
	- label_num_to_disease_map.json
needs to be in the same local directory as the files:
	- dataloader.py
	- epoch_train_val.py
	- main_dl.py
	- models.py
	- naive_model.py
	- plots.py
	- utils.py

################
#naive_model.py#
################
- No arguments
- Contains the code for the majority class prediction model


##########
#plots.py#
##########
- No arguments
- Contains the code for the plots seen in the data exploration section of the report
- feature_map(model, image) plots the feature maps of the model specified by the model parameter of the input image
  specified by the img parameter
- label_histogram(df) plots the distribution of labels as a histogram of the given dataframe df
- plot_class(df, lbl) picks 4 random images from the dataframe df which are of class lbl and plots them

##########
#utils.py#
##########
- No arguments
- Contains the AverageMeter class which is used to store running averages used in training 
  and validation metrics such as loss and accuracy
- Not run

###############
#dataloader.py#
###############
- No arguments
- Contains the main class that is used for loading the data for pytorch
- Not run

####################
#epoch_train_val.py#
####################
- No arguments
- Contains the main functions used for training and validation
- Not run

###########
#models.py#
###########
- No arguments
- Contains the models for the Stacked CNN, ResNet18, ResNet34, ResNet34 (ImageNet pre-trained) and ResNet50
- Not run

############
#main_dl.py#
############
- Has the following arguments:

	- '--net'
		- Type: string
		- Default: 'ResNet34'
		- Description: Which model to use
		- Values: One of 'NetConv', 'ResNet18', 'ResNet34', 'ResNet34Pre', 'ResNet50'
	- '--lr'
		- Type: float
		- Default: 0.01
		- Description: What to set the initial learning rate to
	- '--mom'
		- Type: float
		- Default: 0.9
		- Description: What to set the momentum for SGD to
	- '--patience'
		- Type: int
		- Default: 2
		- Description: What to set the learning rate scheduler number of patience epochs to
	- '--epochs'
		- Type: int
		- Default: 50
		- Description: How many epochs to train for
	- '--ndata'
		- Type: string
		- Default: -1 (use all the data)
		- Description: How many datapoints to use (which are then split into training and validation)
	- '--factor'
		- Type: float
		- Default: 0.2
		- Description: The learning rate decay factor 
	- '--mname'
		- Type: string
		- Default: 'final_model'
		- Description: File name to store the trained model weights in

- Contains the main function that performs all of the training and validation. At the end of training, plots are shown 
  detailing the loss and accuracy per epoch and the final model weights are saved in the local directory with the file
  name specified in the argument '--mname'
- Example usage for training ResNet50 over 5000 datapoints with a learning rate of 0.05 and learning rate scheduler patience
  of 3 with the final model being saved in a file named 'model_weights':
	- python main_dl.py --lr=0.05 --patience=3 --ndata=5000 --mname=model_weights --net=ResNet50

####################
#final_model_85_acc#
####################
- The weights of the final ResNet34 model
- Load as such:
	- import torch
	- from models import ResNet34
	- model = ResNet34(5)
	- model.load_state_dict(torch.load('final_model_85_acc', map_location='cpu')
