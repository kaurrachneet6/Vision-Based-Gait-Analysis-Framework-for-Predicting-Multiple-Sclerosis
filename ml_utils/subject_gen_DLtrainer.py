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
from ml_utils.DLutils import save_model, load_model, design

class GaitTrainer():
    def __init__(self, parameter_dict, hyperparameter_grid, config_path):
        self.parameter_dict = parameter_dict
        self.data_path = self.parameter_dict['path'] + self.parameter_dict['data_path']
        self.labels_file = self.parameter_dict['path'] + self.parameter_dict['labels_file']    
        self.labels = pd.read_csv(self.labels_file, index_col = 0)
        self.framework = self.parameter_dict['framework'] #Defining the framework of interest
        #If "comparision" keyword exists, then we need to keep only common PIDs between the frameworks to compare 
        self.comparision_frameworks = self.parameter_dict['comparision_frameworks']
        self.scenario = self.parameter_dict['scenario']
        self.hyperparameter_grid = hyperparameter_grid
        self.save_results_path = self.parameter_dict['results_path'] + self.parameter_dict['model_path'] + self.framework + '\\'
        self.save_results_prefix = self.parameter_dict['prefix_name'] + '_'
        self.save_results = self.parameter_dict['save_results']
        self.config_path = config_path
        self.re_train_epochs = self.parameter_dict['re_train_epochs']
        
        
    def keep_common_PIDs(self):
        '''
        Since we need to compare across sub-frameworks, we must have same subjects across all sub-frameworks 
        Hence, if there are some subjects that are present in the one sub-frameworks, say W but not in another and vice versa, we eliminate those 
        subjects to have only common subjects acrossall subframeworks. 
        Arguments: 
            data: labels.csv file with all consolidated information,
            frameworks: list of frameworks for which common PIDs need to be filtered
        Returns: 
            common_pids: list of common PIDs/subjects across all the frameworks 
        '''
    
        original_pids = {} #Dictionary with original number of PIDs in each framework (task)
        for framework in self.comparision_frameworks:
            #Appending the original PIDs for each task
            original_pids[framework] = self.labels[self.labels.scenario==framework].PID.unique()
            print ('Original number of subjects in task', framework, 'are:', len(original_pids[framework]))

        #List of common PIDs across all frameworks
        self.common_pids = set(original_pids[self.comparision_frameworks[0]])
        for framework in self.comparision_frameworks[1:]:
            self.common_pids.intersection_update(original_pids[framework])
        self.common_pids = list(self.common_pids)
        print ('Common number of subjects across all frameworks: ', len(self.common_pids))
        print ('Common subjects across all frameworks: ', self.common_pids)
    
    
    def get_data_loaders(self):
        '''
        To define the training and testing data loader to load X, y for training and testing sets in batches 
        Arguments:
            pids_retain_train: Based on the framework, PIDs to extract training data for 
            pids_retain_test: Based on the framework, PIDs to extract testing data for 
            parameter_dict: dictionary with parameters defined 
            train_framework: task to train on 
            test_framework: task to test on 
            normalize_type = 'z' for z-score normalization based on training data mean and std or 'm' for min-max normalization 
                            based on training data min and max 
        Returns: 
            training_data_loader, testing_data_loader to load X, y for training and testing sets in batches 
        '''
        #Subject generalization W or WT framework 
        #Loading the full training data in one go to compute the training data's mean and standard deviation for normalization 
        #We set the batch_size = len(training_data) for the same 
        data = GaitDataset(self.data_path, self.labels_file, self.trial['PID'], framework = self.scenario)   
        data_loader = DataLoader(data, batch_size = len(data), shuffle = self.parameter_dict['shuffle'], \
                                          num_workers = self.parameter_dict['num_workers'])
        #Since we loaded all the training data in a single batch, we can read all data and target in one go
        x_data, target, pid = next(iter(data_loader))
        #Computing the training data mean and standard deviation for z-score normalization of frame count 
        self.train_frame_count_mean_ = torch.mean(data['frame_count'])
        self.train_frame_count_std_ = torch.std(data['frame_count'])
        print ('Training frame count mean: ', self.train_frame_count_mean_)
        print ('Training frame count standard deviation: ', self.train_frame_count_std_)

        #With training data mean/min and standard deviation/max-min computed, 
        #we can load the z-score/min-max normalized training and testing data in batches 
        self.training_data = GaitDataset(self.data_path, self.labels_file, self.pids_retain_train, framework = self.train_framework, \
                                    train_frame_count_mean=self.train_frame_count_mean_, train_frame_count_std=self.train_frame_count_std_)   
        self.testing_data = GaitDataset(self.data_path, self.labels_file, self.pids_retain_test, framework = self.test_framework, \
                                  train_frame_count_mean=self.train_frame_count_mean_, train_frame_count_std=self.train_frame_count_std_) 

        #To make sure the z-score/min-max normalization worked correctly 
        training_data_loader_check = DataLoader(self.training_data, batch_size = len(self.training_data), shuffle = self.parameter_dict['shuffle'], \
                                          num_workers = self.parameter_dict['num_workers'])
        data_check, target_check, pid_check = next(iter(training_data_loader_check))
        #The normalized means for each of the 37 features must be ~0
        print ('Normalized training data\'s mean:', torch.mean(data_check['frame_count']))
        #The normalized standard deviations for each of the 37 feature must be ~1
        print ('Normalized training data\'s standard deviation:', torch.std(data_check['frame_count']))

 




    def subject_gen_setup(self, model_ = None, device_ = torch.device("cuda"), n_splits_ = 5):
        
#         cols_to_drop = ['PID', 'key', 'cohort', 'trial', 'scenario', 'video', 'stride_number', 'label']
#         #Shuffling the cross validation stride data
#         trialW = shuffle(trialW, random_state = 0)
#         #CV for people generalize so no train-test split
#         X = trialW.drop(cols_to_drop, axis = 1)
#         Y = trialW[['PID', 'label']]


#         cols_to_drop = ['PID', 'key', 'cohort', 'trial', 'scenario', 'video', 'stride_number', 'label']

#         #Shuffling the cross validation stride data
#         reduced_data_W = shuffle(reduced_data_W, random_state = 0)
#         #CV for people generalize so no train-test split
#         X_reduced_data_W = reduced_data_W.drop(cols_to_drop, axis = 1)
#         Y_reduced_data_W = reduced_data_W[['PID', 'label']]
        
        if "comparision" in self.framework:
            #Case when we need to retain common subjects in W and WT to compare them
            self.keep_common_PIDs()
            #Retaining the data with only common PIDs
            self.reduced_labels = self.labels[self.labels.PID.isin(self.common_pids)]
            print ('Number of subjects in each cohort in reduced data with common PIDs:\n', \
                   self.reduced_labels.groupby('PID').first()['cohort'].value_counts())
            design()
            #Checking the retained strides in each task after reducing to commpn PIDs only
            for scen in self.comparision_frameworks:
                reduced_labels_scen = self.reduced_labels[self.reduced_labels.scenario==scen].reset_index().drop('index', axis = 1)
                print ('No. of strides retained in scenario', scen, 'are: ', reduced_labels_scen.shape)
                print ('No. of strides retained for each cohort in scenario', scen, 'are:\n', reduced_labels_scen['cohort'].value_counts())
                print ('Imbalance ratio in scenario', scen, '(controls:MS:PD)= 1:X:Y\n', \
                       reduced_labels_scen['cohort'].value_counts()/reduced_labels_scen['cohort'].value_counts()['HOA'])
                design()
            self.trial = self.reduced_labels[self.reduced_labels.scenario==self.scenario].reset_index().drop('index', axis = 1)
               
        else:
            #Case when we can use all subjects in W or WT for full analysis 
            #Subject generalization W/WT framework 
            #Trial W/WT for training and testing both
            self.trial = self.labels[self.labels['scenario']==self.scenario]
            print ('Original number of subjects in trial ', self.scenario, ' for cross validation:', len(self.trial['PID'].unique()))
            print ('Number of subjects in trial ', self.scenario, ' in each cohort:\n', self.trial.groupby('PID').first()['cohort'].value_counts())
            #Total strides and imbalance of labels in the training and testing set
            #Training set 
            print('Strides in trial ', self.scenario, ' W for cross validation: ', len(self.trial))
            print ('HOA, MS and PD strides in trial ', self.scenario, ' :\n', self.trial['cohort'].value_counts())
            print ('Imbalance ratio in trial ', self.scenario, ' (controls:MS:PD)= 1:X:Y\n', self.trial['cohort'].value_counts()/self.trial['cohort'].value_counts()['HOA'])

            
            
            
#         self.get_data_loaders()
#         self.create_folder_for_results()   
    
#         if self.parameter_dict['behavior'] == 'train':
#             self.model = self.create_model(model_, device_)
#             self.X_sl_train = SliceDataset(self.training_data, idx = 0)
#             self.Y_sl_train = SliceDataset(self.training_data, idx = 1)
#             self.PID_sl_train = SliceDataset(self.training_data, idx = 2)
#             self.train(n_splits_)
#             cv_results = pd.DataFrame(self.grid_search.cv_results_)
#             if self.save_results:
#                 cv_results.to_csv(self.save_path+"cv_results.csv")
#             self.learning_curves()
#             print("\nBest parameters: ", self.grid_search.best_params_)
#             if self.save_results:
#                 best_params_file = open(self.save_path + "best_parameters.txt","w") 
#                 best_params_file.write(str(self.grid_search.best_params_))
#                 best_params_file.close() 
#             self.best_model = self.grid_search.best_estimator_
#             if self.save_results:
#                 save_model(self.best_model, self.save_path)
        
#         if self.parameter_dict['behavior'] == 'evaluate':
#             self.training_time = 0
#             self.best_model = load_model(self.save_results_path + self.parameter_dict['saved_model_path'])
# #             print (self.best_model.get_params())
# #             display (pd.DataFrame(self.best_model.history))
        
#         if self.parameter_dict['behavior'] == 'resume_training':
#             self.best_model = load_model(self.save_results_path + self.parameter_dict['saved_model_path'])            
#             self.X_sl_train = SliceDataset(self.training_data, idx = 0)
#             self.Y_sl_train = SliceDataset(self.training_data, idx = 1)
#             self.PID_sl_train = SliceDataset(self.training_data, idx = 2)
#             self.resume_train()
            
        
#         #Count of parameters in the selected model
#         self.total_parameters = sum(p.numel() for p in self.best_model.module.parameters())        
#         self.trainable_params =  sum(p.numel() for p in self.best_model.module.parameters() if p.requires_grad)
#         self.nontrainable_params = self.total_parameters - self.trainable_params

#         self.X_sl_test = SliceDataset(self.testing_data, idx = 0)
#         self.Y_sl_test = SliceDataset(self.testing_data, idx = 1)
#         self.PID_sl_test = SliceDataset(self.testing_data, idx = 2)
    
#         self.evaluate() 
#         self.plot_ROC() 