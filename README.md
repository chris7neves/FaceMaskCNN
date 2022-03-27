# FaceMaskCNN

Get a prediction on what mask a face is wearing in an image!

# Table of Contents
1. [Directory Structure](Directory-Structure)
2. [Installation](Installation)
3. [Notes](Notes)
4. [Use](Use)
5. [Full Run](Full_Run)

**NOTE**: Please go to the [Full Run](Full_Run) section of this README if you simply want to replicate the training, testing and inference procedures followed for our project submission.

## Directory Structure
```
│   .gitignore
│   LICENSE
│   README.md       
├───data
|                
├───image_jsons   
| 
├───notebook
│   |   network_param_helper.ipynb 
|    
├───reports
|     
└───src
    │   augment_data.py
    │   datasets.py
    │   generate_report.py
    │   image_scraper.py
    │   infer.py
    │   main.py
    │   metrics_and_plotting.py
    │   test.py
    │   train.py
    │   util.py
    │   
    ├───configs
    │   |   paths.json
    │   |   paths.py
    └───models
        |   available_models.py
        |   fmcnn1.py
        └───saved_models
            └───1_33_47_73acc.pth
```

**data**: directory that stores the data. Currently, this package wraps only models that are made to predict on the facemask datasets. Future work will make the data paths more easily generalizeable.

**image_jsons**: The directory containing the sources and metadata of all scraped images.

**notebook/network_param_helper.ipynb**: Helper notebook used to more easily determine the parameters of each CNN layer

**reports**: Save location for testing reports generated when specifying the --gen_report flag
<br></br>

**src/augment_data.py**: Script that was writting to generated transformed images used to augment the dataset.

**src/datasets.py**: Creates the pytorch Dataset and Dataloader from data split using train_test_split.

**src/generate_report**: Functions needed to generate a performance report after the model has been run on the test data

**src/image_scraper.py**: Script used to download dataset images from google.

**src/infer.py**: Functions to perform inference on a specified image.

**src/main.py**: The main package entrypoint.

**src/metrics_and_plotting.py**: Helper functions used for determining performance metrics and to generate performance plots.

**src/test.py**: Contains functions used to test the model on a test dataset.

**src/train.py**: Contains code used to run the training loop of the models.

**src/util.py**: Utility and helper functions not related to the model
<br></br>

**configs/paths.json**: Contains the tokenized paths to the image directories

**configs/paths.py**: Contains code used to generate relative paths to project subdirectories.
<br></br>

**models/available_models.py**: Contains code used to assemble the different models, and the criterion, optimizer and transforms used for each.

**models/fmcnn1.py**: Definition of the model Fmcnn1. This is the baseline, best performing model on the dataset.

**models/saved_models/**: Contains the .pth files of all models that have been trained on the masktype dataset. Can be specified in test and infer modes.

## Installation

**This project uses python version 3.7. Other versions of python can be used, however this may have unintended consequences for the user.**

To begin using this project, cd to the FaceMaskCNN directory.

Once in that directory, use the following command to install all packages from the requirements file:


Create the venv for the project (naming it the same as below will ensure that it does not get pushed to the repo):
```
python3 -m venv fcnnvenv
```

Activate the venv:
```
On PC:
Windows CMD;           fcnnvenv\Scripts\activate.bat
Windows PowerShell;    fcnnvenv\Scripts\Activate.ps1

On Unix or MacOS:
source fcnnvenv/bin/activate
```

Install the packages from the requirements.txt:
```
python3 -m pip install -r requirements.txt
```

Make sure that the data folder is in the following structure, after having extracted the Dataset.zip. If any of the following directories do not exist, please create them.

Download the Dataset.zip from this link: https://drive.google.com/file/d/11ee9cGV-W2EbAzFcpP6C38hkGLRXvgjD/view?usp=sharing

```
├───data  
    └──masktype
       └──Dataset
          ├───cloth_mask_TF_aug
          ├───faces_TF_aug
          ├───n95_TF_aug
          ├───n95_valve_TF_aug
          └───n95_valve_TF_aug
```

With each final directory holding all of the image files.

## Notes

Important: Do not change the contents of paths.json. If it needs to be changed for some reason, keep the path ordering the following:

1. "cloth_mask"
2. "faces"
3. "n95"
4. "n95_valve"
5. "procedural_mask"

The order that the image paths appear in the paths.json file dictates the categorical variable that will represent that particular class.
Changing the ordering can have adverse effects on training, testing and inference where the correct class labels are not properly shown.

## Use

**To use any of the following commands, please ensure you are in the FaceMaskCnn/src directory. main.py is the main entrypoint to use the software.**

Note, argparser is implemented for command line manipulation. Running:

```
python main.py -h
```
Will print a list of available commands and arguments.

The FaceMaskCNN project has 4 main commands that can be invoked in the command line. These are:

```
python main.py model_list
```
model_list prints out all of the model definitions ready for training, testing and inference. Note that a model might exist to be trained, but does not have its parameters saved yet. Choosing this model to use during testing or inferrence might yield errors.

```
python main.py train model_name --train_batchsz --val_batchsz --num_epochs --skip_val --save_losses
```
This is the main training entrypoint for the package. 

* model_name: required argument. Specify the name of a model that appears when running 'model_list'
* train_batchsz: the batch size to use during training. Default is 124
* val_batchsz: the batch size to use during validation. Default is 124
* num_epochs: the number of epochs to run the training loop for. Default is 25
* skip_val: do not perform validation during the training phase. More data is used to train the network, but best model saving is disabled. Not recommended.
* save_losses: Save the training-validation loss and accuracy curve to the reports directory

```
python main.py test model_name from_saved --test_batchsz --gen_report
```
This is the main testing entrypoint. Requires a model to have already been trained and its .pth file to be located in the /FaceMaskCNN/src/models/saved_models/ directory.

* model_name: same as above.
* from_saved: the name of the .pth file located in the src/models/saved_models/ directory. This is a filename, not a path.
* test_batchsz: batch size to use for testing. Default is 300.
* gen_report: if specified, generates a .html report of the test statistics of the model, along with a confusion matrix. All is stored in the FaceMaskCNN/reports/ directory

```
python main.py infer img_path model_name from_saved 
```
This is the entry point in order to perform inference on an image. 

* img_path: the path to the image to perform inference on.
* model_name: specify the name of a model that appears when running 'model_list'
* from_saved: the name of the .pth file located in the src/models/saved_models/ directory. This is a filename, not a path.


## Full Run

This section will walk you through a full run of the project from cloning, to training, all the way to testing and inference.

1. Cloning the repository:

Navigate to the location you would like to keep this project in and rnu the command:

```
git clone https://github.com/chris7neves/FaceMaskCNN.git
```

2. Create the data directory 

In the FaceMaskCNN project root create the following directory path:

```
data/masktype/
```

The data and the masktype directories will be empty for now. We will download the dataset in the next step.

3. Download and extract the Dataset.zip file

Navigate to the following google drive link: https://drive.google.com/file/d/11ee9cGV-W2EbAzFcpP6C38hkGLRXvgjD/view?usp=sharing

Download the file that is found there.

Once downloaded, place it in:

```
data/masktype/Dataset.zip
```

Extract the folder. The folder will be extracted to "Dataset/". You need to take the extracted folder and move it to /data/masktype/.

The resulting directory structure of the data/ directory should now be:

```
├───data  
    └──masktype
       └──Dataset
          ├───cloth_mask_TF_aug
          ├───faces_TF_aug
          ├───n95_TF_aug
          ├───n95_valve_TF_aug
          └───n95_valve_TF_aug
```

You can delete any empty folders and files outside of the above data directory structure.

4. Create a virtual environment

On windows, run the command to create a venv:
```
python3 -m venv fcnnvenv
```

or, depending on your python installation, simply:
```
python -m venv fcnnvenv
```

5. Activate the venv

On windows, for powershell (Please see [Installation](Installation) for Unix and MacOS equivalent commands):
```
./fcnnvenv/Scripts/Activate.ps1
```

6. Install required packages from the included requirements.txt

To use pip in order to install requirements, make sure that you currently have your fcnnvenv active and run the following command:

```
python -m pip install -r requirements.txt
```

7. CD into the src directory

```
cd src
```

8. Run the training loop

Our results were obtained by running the training loop for 25 epochs, with a default batchsize that is hardcoded in the code. To reproduce that, we run:

```
python main.py train Fmcnn1 --num_epochs 25 --save_losses
```
This command will train the Fmcnn1 model for 25 epochs, and will save the training/validation loss curve to the FaceMaskCNN/reports/ directory