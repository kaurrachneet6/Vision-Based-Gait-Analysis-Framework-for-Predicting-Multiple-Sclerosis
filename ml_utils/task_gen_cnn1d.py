from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *
from ml_utils.split import StratifiedGroupKFold

def load_model(path):
    '''
    Loads a saved skorch model
    Arguments:
        path: file path to saved skorch model
    Returns:
        Loaded skorch model
    '''
    '''
    net = NeuralNetClassifier(
    module=LSTM,
    criterion=nn.CrossEntropyLoss)

    net.initialize()
    net.load_params(f_params=path+"model.pkl", f_optimizer=path+"opt.pkl", f_history=path+'history.json')
    return net
    '''
    with open(path+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

