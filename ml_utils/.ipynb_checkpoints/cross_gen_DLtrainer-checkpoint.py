from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *
from ml_utils.split import StratifiedGroupKFold
from ml_utils import gait_data_loader
reload(gait_data_loader)
from ml_utils.gait_data_loader import GaitDataset
from ml_utils import DLutils
reload(DLutils)
from ml_utils.DLutils import save_model, load_model, design, custom_StandardScaler

class GaitTrainer():
    def __init__(self, parameter_dict, hyperparameter_grid, config_path):
        self.parameter_dict = parameter_dict
        self.data_path = self.parameter_dict['path'] + self.parameter_dict['data_path']
        self.labels_file = self.parameter_dict['path'] + self.parameter_dict['labels_file']    
        self.labels = pd.read_csv(self.labels_file, index_col = 0)
        self.framework = self.parameter_dict['framework'] #Defining the framework of interest
        self.train_framework = self.parameter_dict['train_framework']
        self.test_framework = self.parameter_dict['test_framework']
        self.hyperparameter_grid = hyperparameter_grid
        self.save_results_path = self.parameter_dict['results_path']  + self.framework + '\\'+ self.parameter_dict['model_path']
        self.save_results_prefix = self.parameter_dict['prefix_name'] + '_'
        self.save_results = self.parameter_dict['save_results']
        self.config_path = config_path
        self.re_train_epochs = self.parameter_dict['re_train_epochs']
        
        
    def extract_train_test_common_PIDs(self):
        '''
        Since we need to train on some subjects of W/VBW and test on remaining subjects of WT/VBWT respectively, we keep subjects that 
        only belong to W/VBW and not to WT/VBWT as only training subjects and similarly subjects that belong to only WT/VBWT as only testing 
        subjects. Further, for common subjects in training and testing, we use 5-fold cross validation. 

        For implementation of task+subject generalization together framework 1: i.e. train on some subjects in W and test on 
        remaining separate set of subjects in WT,  since we have 32 subjects in training/W and 26 subjects in testing/WT and 
        25 subjects that are common in both W and WT. We always keep the (32-25) = 7 subjects only available in W in training and 
        always keep (26-25) = 1 subject only available in WT in testing along with cross validation folds created for training and 
        testing sets from the 25 common subjects in both. So, basically, for 5 fold cross validation on 25 common subjects, we train 
        on 20 + 7 subjects and test on 5+1 subjects where these 20 and 5 subjects keep on changing with each fold, but the 7 and 1 
        subjects remain the same.
        Arguments: 
            data: labels.csv file with all consolidated information,
            train_framework: Training framework
            test_framework: Testing framework
        Returns: 
            train_pids: list of PIDs/subjects that are only included in the training set 
            test_pids: list of PIDs/subjects that are only included in the test set
            common_pids: list of common PIDs/subjects across the training and test set to be used for CV 
        '''

        original_pids = {} #Dictionary with original number of PIDs in each framework (task)
        #Appending the original PIDs for each task
        original_pids[self.train_framework] = self.labels[self.labels.scenario==self.train_framework].PID.unique()
        original_pids[self.test_framework] = self.labels[self.labels.scenario==self.test_framework].PID.unique()
        print ('Original number of subjects in training task', self.train_framework, 'are:', len(original_pids[self.train_framework]))
        print ('Original number of subjects in testing task', self.test_framework, 'are:', len(original_pids[self.test_framework]))

        #List of common PIDs across the train and test frameworks
        self.common_pids = list(set(original_pids[self.train_framework]) & set(original_pids[self.test_framework]))
        print ('Common number of subjects across train and test frameworks: ', len(self.common_pids))
        print ('Common subjects across train and test frameworks: ', self.common_pids)
        #List of PIDs only in the training set but not in the test set
        self.train_pids = list(set(original_pids[self.train_framework])^set(self.common_pids))
        print ('Number of subjects only in training framework: ', len(self.train_pids))
        print ('Subjects only in training framework: ', self.train_pids)
        #List of PIDs only in the testing set but not in the training set
        self.test_pids = list(set(original_pids[self.test_framework])^set(self.common_pids))
        print ('Number of subjects only in test framework: ', len(self.test_pids))
        print ('Subjects only in test framework: ', self.test_pids)    
        


    def cross_gen_setup(self, model_ = None, device_ = torch.device("cuda"), n_splits_ = 5):
        self.extract_train_test_common_PIDs()
        design()
        
        #Trial W for training 
        self.trial_train = self.labels[self.labels['scenario']==self.train_framework] #Full trial W with all 32 subjects 
        #Trial WT for testing 
        self.trial_test = self.labels[self.labels['scenario']==self.test_framework] #Full trial WT with all 26 subjects 

        #Full training data stats 
        print ('Number of subjects in trial W in each cohort:\n', self.trial_train.groupby('PID').first()['cohort'].value_counts())
        print('Strides in complete training set: ', len(self.trial_train))
        print ('HOA, MS and PD strides in complete training set:\n', self.trial_train['cohort'].value_counts())
        design()

        #Full testing data stats 
        print ('Number of subjects in trial WT in each cohort:\n', self.trial_test.groupby('PID').first()['cohort'].value_counts())
        print('Strides in complete testing set: ', len(self.trial_test))
        print ('HOA, MS and PD strides in complete testing set:\n', self.trial_test['cohort'].value_counts())
        design()

        #Training only data with strides from W
        train_only_trial_train = self.trial_train[self.trial_train.PID.isin(self.train_pids)] #subset of trial W with subjects only present in trial W but not in trial WT
        print ('Number of subjects only in trial W in each cohort:\n', train_only_trial_train.groupby('PID').first()['cohort'].value_counts())
        print('Strides of subjects only in trial W: ', len(train_only_trial_train))
        print ('HOA, MS and PD strides in of subjects only in trial W :\n', train_only_trial_train['cohort'].value_counts())
        design()

        #Testing only data with strides from WT
        test_only_trial_test = self.trial_test[self.trial_test.PID.isin(self.test_pids)] #subset of trial WT with subjects only present in trial WT but not in trial W
        print ('Number of subjects only in trial WT in each cohort:\n', test_only_trial_test.groupby('PID').first()['cohort'].value_counts())
        print('Strides of subjects only in trial WT: ', len(test_only_trial_test))
        print ('HOA, MS and PD strides in of subjects only in trial WT :\n', test_only_trial_test['cohort'].value_counts())
        design()

        #Training data with strides from W for common PIDs in trials W and WT
        self.train_trial_train_commonPID = self.trial_train[self.trial_train.PID.isin(self.common_pids)] #subset of trial W with common subjects in trial W and WT
        print ('Number of subjects common to trials W and WT in each cohort:\n', self.train_trial_train_commonPID.groupby('PID').first()['cohort'].value_counts())
        print('Strides in trial W in each cohort of subjects common to trials W and WT: ', len(self.train_trial_train_commonPID))
        print ('HOA, MS and PD strides in trial W of subjects common to trials W and WT:\n', self.train_trial_train_commonPID['cohort'].value_counts())
        design()

        #Testing data with strides from WT for common PIDs in trials W and WT
        self.test_trial_test_commonPID = self.trial_test[self.trial_test.PID.isin(self.common_pids)] #subset of trial W with common subjects in trial W and WT
        print ('Number of subjects common to trials W and WT in each cohort:\n', self.test_trial_test_commonPID.groupby('PID').first()['cohort'].value_counts())
        print('Strides in trial WT in each cohort of subjects common to trials W and WT: ', len(self.test_trial_test_commonPID))
        print ('HOA, MS and PD strides in trial WT of subjects common to trials W and WT:\n', self.test_trial_test_commonPID['cohort'].value_counts())
        design()

        