'''
## Gait Video Study 
### 1D Convolutional neural network (CNN) on subject generalization frameworks, namely a) W, b) WT, c) VBW and d) VBWT,  to classify HOA/MS/PD strides and subjects using cross validation 
#### Remember to add the original count of frames in a single stride (before down sampling via smoothing) for each stride as an additional artificial feature to add information about speed of the subject to the model
1. Save the optimal hyperparameters, confusion matrices and ROC curves for each algorithm.
2. Make sure to not use x, y, z, confidence = 0, 0, 0, 0 as points for the model since they are simply missing values and not data points, so make sure to treat them before inputting to model 
3. Make sure to normalize (z-score normalization) the features before we feed them to the model.
4. We use Group 5-fold stratified cross validation for evaluation.
5. Compare 1D CNN among the 4 sub-frameworks of subject generalization by retaining only common subjets across the 4 frameworks.
6. Plot the training and testing loss vs. epochs and maybe training and testing accuracy vs. epochs. 
7. Try both z-score and min-max normalizations. Compute the training data min (using bottom 1 percentile of training data ) and max (using top 1 percentile of training data) for min-max normalization to avoid extreme outliers influencing the min/max value for normalization.
8. Try appending speed of stride in the begining along with 20x36 grid of body coordinates to process the label and appending speed after 20x36 grid is processed to a feature set represeting body coordinates and then using linear layers to process the label 
'''

from importlib import reload
from ml_utils.imports import *

from ml_utils import subject_gen_DLtrainer, DLutils, cnn1d_model, gait_data_loader
reload(subject_gen_DLtrainer)
reload(DLutils)
reload(cnn1d_model)
from ml_utils.DLutils import set_random_seed, accuracy_score_multi_class
from ml_utils.subject_gen_DLtrainer import GaitTrainer
from ml_utils.cnn1d_model import CNN1D
from ml_utils.gait_data_loader import GaitDataset

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
model_class_ = CNN1D   
model_ = CNN1D (in_chans, out_chans, dropout)

trainer = GaitTrainer(parameter_dict, hyperparameter_grid, config_path = args.config_path)
trainer.subject_gen_setup(model_class_, model_, device_ = device, n_splits_ = 2)