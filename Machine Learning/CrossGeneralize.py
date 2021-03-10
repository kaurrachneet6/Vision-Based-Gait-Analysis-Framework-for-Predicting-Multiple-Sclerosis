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

from ml_utils import cross_gen_DLtrainer, DLutils, CNN1d_model, gait_data_loader, TCN_model, LSTM_model, GRU_model, RNN_model, MULTISCALE_RESNET_model, RESNET_model
reload(cross_gen_DLtrainer)
reload(DLutils)
reload(CNN1d_model)
reload(TCN_model)
reload(LSTM_model)
reload(GRU_model)
reload(RNN_model)
reload(RESNET_model)
reload(MULTISCALE_RESNET_model)
from ml_utils.DLutils import set_random_seed, accuracy_score_multi_class
from ml_utils.cross_gen_DLtrainer import GaitTrainer
from ml_utils.CNN1d_model import CNN1D
from ml_utils.TCN_model import TCN
from ml_utils.LSTM_model import LSTM
from ml_utils.GRU_model import GRU
from ml_utils.RNN_model import RNN
from ml_utils.MULTISCALE_RESNET_model import MSResNet
from ml_utils.RESNET_model import ResNet
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


if parameter_dict["model"]=="CNN1D":
    in_chans = 36
    out_chans = [64, 128, 64]
    kernel_size = [8, 5, 3]
    stride = [1, 1, 1]
    dilation = [1, 1, 1]
    groups = [1, 1, 1]
    batch_norm = [True, False, False]
    dropout = [0.3, 0, 0]
    maxpool = [False, False, True]
    maxpool_kernel_size = [2, 2, 2]
    dense_out_sizes = [10]
    dense_pool = False
    dense_pool_kernel_size= 2
    dense_dropout = [0]
    global_average_pool = False
    num_classes = 3     
    time_steps = 20 #Number of time steps in one data sample
    position_encoding = True
    model_ = CNN1D(in_chans, out_chans, kernel_size, stride, dilation, groups, batch_norm, dropout, maxpool, maxpool_kernel_size, dense_out_sizes, dense_pool, dense_pool_kernel_size, dense_dropout, global_average_pool, num_classes, time_steps, position_encoding)

if parameter_dict["model"] == "Resnet":
    in_chans = 36
    initial_conv_layer = True
    block_name = 'basic_block' #'bottleneck' 
    layers = [1, 1, 1, 0]
    kernel_size_conv1 = 1 
    kernel_size_conv2 = 3 
    kernel_size_conv3 = 1, 
    stride_layer64 = [1, 1, 1] 
    stride_layer128 = [1, 2, 1]
    stride_layer256 = [1, 2, 1]
    stride_layer512 = [1, 2, 1]  
    position_encoding = False
    num_classes=3
    model_ = ResNet(in_chans, initial_conv_layer, block_name, layers, kernel_size_conv1, kernel_size_conv2, kernel_size_conv3, stride_layer64, stride_layer128, stride_layer256, stride_layer512, position_encoding, num_classes)
    
if parameter_dict["model"] == "MSResnet":
    in_chans = 36
    layers=[1, 1, 1, 1]
    num_classes = 3
    model_ = MSResNet(in_chans, layers, num_classes) 
    
if parameter_dict["model"]=="TCN":
    in_chans = 36 
    out_chans = 3
    num_channels = [20]*2
    kernel_size = 3
    dropout = 0.3
    model_ = TCN(in_chans, out_chans, num_channels, kernel_size, dropout) 

if (parameter_dict["model"]=="LSTM") or (parameter_dict["model"]=="GRU") or (parameter_dict["model"]=="RNN"):
    in_chans = 36 #36 body coordinate features 
    hidden_size1 = 30
    num_layers1 = 3
    hidden_size2 = 20
    num_layers2 = 2
    num_classes = 3
    dropout = 0.3
    bidirectional = False
    pre_out = 50
    single_layer = False
    linear_size = 1 #Default is 1 for a single FC layer after the LSTM layers
    use_layernorm = False
    hyperparameter_grid['net__module__batch_size'] = hyperparameter_grid['net__batch_size']
    hyperparameter_grid['net__module__device1'] = [device]
    
    if parameter_dict["model"]=="LSTM":
        model_ = LSTM(in_chans, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional, pre_out, single_layer, linear_size, use_layernorm, hyperparameter_grid['net__module__batch_size'][0], hyperparameter_grid['net__module__device1'][0])   
    
    if parameter_dict["model"]=="GRU":
        model_ = GRU(in_chans, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional, pre_out, single_layer, linear_size, use_layernorm, hyperparameter_grid['net__module__batch_size'][0], hyperparameter_grid['net__module__device1'][0])   
    
    if parameter_dict["model"]=="RNN":
        model_ = RNN(in_chans, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional, pre_out, single_layer, linear_size, use_layernorm, hyperparameter_grid['net__module__batch_size'][0], hyperparameter_grid['net__module__device1'][0])   

trainer = GaitTrainer(parameter_dict, hyperparameter_grid, config_path = args.config_path)
trainer.cross_gen_setup(model_, device_ = device, n_splits_ = 5, datastream = parameter_dict["datastream"])
