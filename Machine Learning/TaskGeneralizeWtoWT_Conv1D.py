'''
## Gait Video Study 
### 1D Convolutional neural network (CNN) on task generalization framework 1: train on walking (W) and test on walking while talking (WT) to classify HOA/MS/PD strides and subjects 
#### Remember to add the original count of frames in a single stride (before down sampling via smoothing) for each stride as an additional artificial feature to add information about speed of the subject to the model

1. Save the optimal hyperparameters, confusion matrices and ROC curves for each algorithm.
2. Make sure to not use x, y, z, confidence = 0, 0, 0, 0 as points for the model since they are simply missing values and not data points, so make sure to treat them before inputting to model 
3. Make sure to normalize (z-score) the features before we feed them to the model.
4. Make sure to set a random seed wherever required for reproducible results.
5. Plot the training and testing loss vs. epochs and maybe training and testing accuracy vs. epochs. 
6. Try both z-score and min-max normalizations. Compute the training data min (using bottom 1 percentile of training data ) and max (using top 1 percentile of training data) for min-max normalization to avoid extreme outliers influencing the min/max value for normalization.
7. Try appending speed of stride in the begining along with 20x36 grid of body coordinates to process the label and appending speed after 20x36 grid is processed to a feature set represeting body coordinates and then using linear layers to process the label.
Refer https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/51
8. Try adding the positional encoding on the frames to tell the network where we are on the stride (at the beginning, middle or at the end) during the processing 
9. For parallel grid search using skorch, use https://skorch.readthedocs.io/en/stable/user/parallelism.html

For skorch: Refer https://github.com/skorch-dev/skorch/issues/524, https://github.com/skorch-dev/skorch/blob/master/docs/user/FAQ.rst
'''
            
from importlib import reload
from ml_utils.imports import *

from ml_utils import task_gen_DLtrainer, DLutils, CNN1d_model
reload(task_gen_DLtrainer)
reload(DLutils)
reload(CNN1d_model)
from ml_utils.DLutils import set_random_seed
from ml_utils.task_gen_DLtrainer import GaitTrainer
from ml_utils.CNN1d_model import CNN1D

#Set up vars for parsing
hyperparameter_grid = {}
parameter_dict = {}

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, help='Path to config file')
args = parser.parse_args()

#Load config
with open(args.config_path) as f: 
    config_data = f.read()

config = json.loads(config_data)

#Parse through imported dictionary
for key, value in config.items():
    if 'param_' in key:
        pkey = key.replace("param_", "")
        if 'optimizer' in pkey:
            op = []
            for optim_string in value:
                op.append(optims[optim_string])
            hyperparameter_grid[pkey] = op
        else:
            hyperparameter_grid[pkey] = value
    else:
        parameter_dict[key] = value
        
        
use_cuda = torch.cuda.is_available() #use_cuda is True if cuda is available 
set_random_seed(0, use_cuda) #Setting a fixed random seed for reproducibility 
device = torch.device("cuda" if use_cuda else "cpu")

in_chans = 20
out_chans = 3
dropout = 0.3
model_ = CNN1D(in_chans, out_chans, dropout) 
                       
trainer = GaitTrainer(parameter_dict, hyperparameter_grid, config_path = args.config_path)
trainer.task_gen_setup(model_, device_ = device, n_splits_ = 2)

#To do/check
#argparse pipeline - Done
#Reproducibility - Check on HAL 
#Shuffling - Done
#Saved model, when loaded back, starts training from where it left from, and/or produces the same results on 
#evaluate as the original saved model - Done