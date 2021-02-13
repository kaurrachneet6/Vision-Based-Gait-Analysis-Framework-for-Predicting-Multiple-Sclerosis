'''
## Gait Video Study 
### 1D Convolutional neural network (CNN) on task+subject generalization together frameworks, namely a) train on some subjects in W-> test on separate set of subjects in WT and b) train on some subjects in VBW-> test on separate set of subjects in VBWT to classify HOA/MS/PD strides and subjects 
#### Remember to add the original count of frames in a single stride (before down sampling via smoothing) for each stride as an additional artificial feature to add information about speed of the subject to the model

1. Save the optimal hyperparameters, confusion matrices and ROC curves for each algorithm.
2. Make sure to not use x, y, z, confidence = 0, 0, 0, 0 as points for the model since they are simply missing values and not data points, so make sure to treat them before inputting to model 
3. Make sure to normalize (z-score normalization) the features before we feed them to the model.
4. For implementation of task+subject generalization together framework 1: i.e. train on some subjects in W and test on remaining separate set of subjects in WT,  since we have 32 subjects in training/W and 26 subjects in testing/WT and 25 subjects that are common in both W and WT. We always keep the (32-25) = 7 subjects only available in W in training and always keep (26-25) = 1 subject only available in WT in testing along with cross validation folds created for training and testing sets from the 25 common subjects in both. So, basically, for 5 fold cross validation on 25 common subjects, we train on 20 + 7 subjects and test on 5+1 subjects where these 20 and 5 subjects keep on changing with each fold, but the 7 and 1 subjects remain the same.
5. We use stratified group 5 fold cross validation.
'''

from importlib import reload
from ml_utils.imports import *

from ml_utils import cross_gen_DLtrainer, DLutils, CNN1d_model, gait_data_loader
reload(cross_gen_DLtrainer)
reload(DLutils)
reload(CNN1d_model)
from ml_utils.DLutils import set_random_seed, accuracy_score_multi_class
from ml_utils.cross_gen_DLtrainer import GaitTrainer
from ml_utils.CNN1d_model import CNN1D
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
model_ = CNN1D (in_chans, out_chans, dropout)

trainer = GaitTrainer(parameter_dict, hyperparameter_grid, config_path = args.config_path)
trainer.cross_gen_setup(model_, device_ = device, n_splits_ = 2)