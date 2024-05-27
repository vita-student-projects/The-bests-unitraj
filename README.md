# Modifications from original unitraj repository 
https://github.com/vita-epfl/unitraj-DLAV.git

### Milestone 1
- The ptr.py file containing the network model has been filled
- Few modifications for experiences has been done on the configuration files
- In train.py file, the validation metric is replace by minADE6 for our expeminents

### Milestone 2
- One new parameter `residual` in the `ptr.yaml` file
- Added residuals for encoder layers in `ptr.py` model

- One new parameter `augment` in the `config.yaml` file
- Added random masking while training in `__getitem__` method of `BaseDataset`class in the `base_dataset.py` file. This augmentation is done in a new method named `random_zero_out`.

### Milestone 3
- A new model `gameformer` is added to the unitraj Framework based on [_1_](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.pdf) and there [code](https://github.com/MCZhi/GameFormer/tree/main). This method is based on game theorie to perform the best prediction possible.

#### 3.1 Gameformer dataset
- The gameformer dataset remains the same as for ptr model

#### 3.2 GameFormer model
- The model [`gameformer.py`](motionnet/models/gameformer/gameformer.py) is mainly composed in three parts. The encoder, the multilevel decoder and the best level selection.
##### 3.2.1 Encoder
- The encoder proposed by [_1_](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.pdf) is not well adapted for our dataset since it requires map lines from each vehicle point of view but we only have the overall map.
- We choose instead the ptr encoder model which encode the scene from ego point of view and we reproduce the with neighbors.
- Note: the ptr encoder worked well if all agents/map points are translate/rotate arround the z axis in order at the last time step ego is at position (0,0) with a heading of pi/2. Hence we perform some agents / map rotation each time we want to predict a neighbor trajectory.

##### 3.2.2 Decoder
- The decoder is mainly the same as proposed by [_1_](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.pdf).
- There is first an initial stage to predict trajectories only from past features.
- Then there are k level interactions decoder to modify trajectory predictions based on future interactions.

##### 3.2.3 Best level selection
- Based on the scores of all levels, we would like to select the level with highest score for every prediction.
- IMPORTANT NOTE : We were not able to train the scores correctly during this short project time slot hence the predictor actually act as a random selector.

#### 3.3 GameFormer modules
- You can find all modules needed by gameformer module in the [`gameformer_mudules.py`](motionnet/models/gameformer/gameformer_modules.py) file.
- The only change is in the loss computation. Instead of computing the loss only on ego as ptr, we compute a mean loss for all agents prediction at each decoder level. And we add a score loss to train scores to inversaly fit the minADE (again the in out actual implementation, the scores are not well trained)

#### 3.4 Gameformer config
- In the [`gameformer.yaml`](motionnet/configs/method/gameformer.yaml) config file there are few more parameters as in the ptr method.
- You can set the number of levels, the score loss weight and the output desired level (set to -1 for scores based prediction).
- It is also possible to train only the decoder parts of the model by using an pretrained ptr encoder



# UniTraj

**A Unified Framework for Cross-Dataset Generalization of Vehicle Trajectory Prediction**

UniTraj is a framework for vehicle trajectory prediction, designed by researchers from VITA lab at EPFL. 
It provides a unified interface for training and evaluating different models on multiple dataset, and supports easy configuration and logging. 
Powered by [Hydra](https://hydra.cc/docs/intro/), [Pytorch-lightinig](https://lightning.ai/docs/pytorch/stable/), and [WandB](https://wandb.ai/site), the framework is easy to configure, train and logging.
In this project, you will be using UniTraj to train and evalulate a model we call PTR (predictive transformer) on the data we have given to you.

## Installation

First start by cloning the repository:
```bash
git clone https://github.com/vita-epfl/unitraj-DLAV.git
cd unitraj-DLAV
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate

# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

Finally, install Unitraj and login to wandb via:
```bash
cd unitraj-DLAV # Go to the folder you cloned the repo
pip install -r requirements.txt
pip install -e .
wandb login
```
If you don't have a wandb account, you can create one [here](https://wandb.ai/site). It is a free service for open-source projects and you can use it to log your experiments and compare different models easily.


You can verify the installation of UniTraj via running the training script:
```bash
python train.py method=ptr
```
The incomplete PTR model will be trained on several samples of data available in `motionnet/data_samples`.

## Code Structure
There are three main components in UniTraj: dataset, model and config.
The structure of the code is as follows:
```
motionnet
├── configs
│   ├── config.yaml
│   ├── method
│   │   ├── ptr.yaml
├── datasets
│   ├── base_dataset.py
│   ├── ptr_dataset.py
├── models
│   ├── ptr
│   ├── base_model
├── utils
```
There is a base config, dataset and model class, and each model has its own config, dataset and model class that inherit from the base class.

## Data
You can access the data [here](https://drive.google.com/file/d/1mBpTqM5e_Ct6KWQenPUvNUBJWHn3-KUX/view?usp=sharing). For easier use on SCITAS, we have also provided the dataset in scitas on `/work/vita/datasets/DLAV_unitraj`. We have provided a train and validation set, as well as three testing sets of different difficulty levels: easy, medium and hard.
You will be evaluating your model on the easy test set for the first milestone, and the medium and hard test sets for the second and third milestones, respectively. 
Don't forget to put the path to the real data in the config file.


## Your Task
Your task is to complete the PTR model and train it on the data we have provided. 
The model is a transformer-based model that takes the past trajectory of the vehicle and its surrounding agents, along with the map, and predicts the future trajectory.
![system](https://github.com/vita-epfl/unitraj-DLAV/blob/main/docs/assets/PTR.png?raw=true)
This is the architecture of the encoder part of model (where you need to implement). Supposing we are given the past t time steps for M agents and we have a feature vector of size $d_K$ for each agent at each time step, the encoder part of the model consists of the following steps:
1. Add positional encoding to the input features at the time step dimension for distinguish between different time steps.
2. Perform the temporal attention to capture the dependencies between the trajectories of each agent separately.
3. Perform the spatial attention to capture the dependencies between the different agents at the same time step.
These steps are repeated L times to capture the dependencies between the agents and the time steps.

The model is implemented in `motionnet/models/ptr/ptr_model.py` and the config is in `motionnet/configs/method/ptr.yaml`. 
Take a look at the model and the config to understand the structure of the model and the hyperparameters.

You are asked to complete three parts of the model in `motionnet/models/ptr/ptr_model.py`:
1. The `temporal_attn_fn` function that computes the attention between the past trajectory and the future trajectory.
2. The `spatial_attn_fn` function that computes the attention between different agents at the same time step.
3. The encoder part of the model in the `_forward` function. 

You can find the instructions and some hints in the file itself. 

## Submission and Visualization
You could follow the steps in the [easy kaggle competition](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-2024/overview) to submit your results and compare them with the other students in the leaderboard.
Here are the [medium](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-medium/overview) and [hard](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-hard/overview) competitions for the second and third milestones, respectively.
We have developed a submission script for your convenience. You can run the following command to generate the submission file:
```bash
python generate_predictions.py method=ptr
```
Before running the above command however, you need to put the path to the checkpoint of your trained model on the config file under `ckpt_path`. You can find the checkpoint of your trained model in the `lightning_logs` directory in the root directory of the project. 
For example, if you have trained your model for 10 epochs, you will find the checkpoint in `lightning_logs/version_0/checkpoints/epoch=10-val/brier_fde=30.93.ckpt`. You need to put the path to this file in the config file.

Additionally, for the `val_data_path` in the config file, you need to put the path to the test data you want to evaluate your model on. For the easy milestone, you can put the path to the easy test data, and for the second and third milestones, you can put the path to the medium and hard test data, respectively.

The script will generate a file called `submission.csv` in the root directory of the project. You can submit this file to the kaggle competition. As this file could be big, we suggest you to compress it before submitting it.

In addition, the script will make some visualizations of the predictions and save them in the `visualizations` directory in the root directory of the project. You can take a look at these visualizations to understand how your model is performing.
It's also needed for the report of the first milestone, so don't forget to include some of them in your report.
