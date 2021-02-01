from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *

def set_random_seed(seed_value, use_cuda):
    '''
    To set the random seed for reproducibility of results 
    Arguments: seed value and use cuda (True if cuda is available)
    '''
    random.seed(seed_value)
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)
#     torch.set_deterministic(True)
    if use_cuda: 
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        

def load_model(path):
    '''
    Loads a saved skorch model
    Arguments:
        path: file path to saved skorch model
    Returns:
        Loaded skorch model
    '''
    with open(path+'model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, lpath):
    '''
    Saves a skorch model
    Arguments:
        model: Skorch model
        lpath: file path to save the skorch model
    '''
    
    #loc = save_model_destination+str(strides_per_sequence)+"strides/"+time+dataset+framework+str(bidirectional)
    #model.save_params(f_params=loc+'model.pkl', f_optimizer=loc+'opt.pkl', f_history=loc+'history.json')
    
    with open(lpath+"model.pkl", 'wb') as f:
        pickle.dump(model, f)


def accuracy_score_multi_class(net, X, y):
    '''
    Function to compute the accuracy using the softmax probabilities predicted via skorch neural net
    Arguments:
        net: skorch model
        X: data 
        y: true target labels
    Returns:
        accuracy 
    '''
    y_pred = net.predict(X)
    y_pred_label = y_pred.argmax(axis = 1)
#     print ('y_pred_label', y_pred_label, y_pred_label.shape)
#     print ('y_true', y, y.shape)
    return accuracy_score(y, y_pred_label)


def design():
    print ('******************************************')