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
    
    
class custom_StandardScaler():
    '''
    Class for custom standard scalar to z-score normalize the frame count using training folds for training and testing folds 
    '''
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        X_frame_count = [X[idx]['frame_count'] for idx in range(len(X))]
        X_frame_count = np.array([X_frame_count]).T
        self.scaler.fit(X_frame_count)
#         print ('Mean and var', self.scaler.mean_, self.scaler.var_)
        return self

    def transforms(self, x):
        x['frame_count'] = self.scaler.transform(np.array([[x['frame_count']]]).T)[0][0]
        return x

    def transform(self, X, y=None):
        X.transform = self.transforms
        return X
    

def accuracy_score_multi_class_cv(net, X, y):
        '''
        Function to compute the accuracy using the softmax probabilities predicted via skorch neural net in a cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            accuracy 
        '''
        y_pred = net.predict(X)
        print ('y_pred: ', y_pred)
        y_true_label = [int(y_val) for y_val in y]
        print ('y true label: ', y_true_label)
        y_pred_label = y_pred.argmax(axis = 1)
        self.yoriginal.append(y_true_label)
        self.ypredicted.append(y_pred_label)
    #     print ('y_pred_label', y_pred_label, y_pred_label.shape)
        print ('current self.yoriginal: ', self.yoriginal)
    
        accuracy = accuracy_score(y_true_label, y_pred_label)
        precision_macro = precision_score(y_true_label, y_pred_label, average = 'macro')
        precision_micro = precision_score(y_true_label, y_pred_label, average = 'micro')
        precision_weighted = precision_score(y_true_label, y_pred_label, average = 'weighted')
        recall_macro = recall_score(y_true_label, y_pred_label, average = 'macro')
        recall_micro = recall_score(y_true_label, y_pred_label, average = 'micro')
        recall_weighted = recall_score(y_true_label, y_pred_label, average = 'weighted')        
        f1_macro = f1_score(y_true_label, y_pred_label, average = 'macro')
        f1_micro = f1_score(y_true_label, y_pred_label, average = 'micro')
        f1_weighted = f1_score(y_true_label, y_pred_label, average = 'weighted') 
        auc_macro = roc_auc_score(y_true_label, y_pred, average = 'macro', multi_class = 'ovo')
        auc_weighted = roc_auc_score(y_true_label, y_pred, average = 'weighted', multi_class = 'ovo')
        scores = {'accuracy': accuracy, 'precision_macro': precision_macro, 'precision_micro':precision_micro, 'precision_weighted': precision_weighted, 'recall_macro': recall_macro, 'recall_micro': recall_micro, 'recall_weighted': recall_weighted, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_weighted, 'auc_macro': auc_macro, 'auc_weighted': auc_weighted}
        print ('current scores: ', scores)
        return scores