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
        #If "comparision" keyword exists, then we need to keep only common PIDs between the frameworks to compare 
        self.comparision_frameworks = self.parameter_dict['comparision_frameworks']
        self.scenario = self.parameter_dict['scenario']
        self.hyperparameter_grid = hyperparameter_grid
        self.save_results_path = self.parameter_dict['results_path']  + self.framework + '\\'+ self.parameter_dict['model_path']
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
    
    
    def create_folder_for_results(self):
        #Create folder for saving results
        time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
        self.save_path = self.save_results_path + self.save_results_prefix + time_now+"\\"
        print("save path: ", self.save_path)
        os.mkdir(self.save_path)
        #Copy config file to results folder
        with open(self.config_path, 'rb') as src, open(self.save_path+"config.json", 'wb') as dst: dst.write(src.read())

    
    def create_model(self, model, device_):
        '''
        Creates Skorch model
        Arguments:
            device: cuda if GPU is available or cpu 
        Returns:
            Created skorch network
        '''
        net = NeuralNet(
            model,
            max_epochs = 2,
            lr = .0001,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            device = device_,
            iterator_train__shuffle=True,
            batch_size= -1, #Batch size = -1 means full data at once 
            callbacks=[EarlyStopping(patience = 100, lower_is_better = True, threshold=0.0001), 
            #('lr_scheduler', LRScheduler(policy=torch.optim.lr_scheduler.StepLR, step_size = 500)),
            (EpochScoring(scoring=DLutils.accuracy_score_multi_class, lower_is_better = False, on_train = True, name = "train_acc")),
            (EpochScoring(scoring=DLutils.accuracy_score_multi_class, lower_is_better = False, on_train = False, name = "valid_acc"))
                      ]
        )
        return net
            
    
    def accuracy_score_multi_class_cv(self, net, X, y):
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
        print ('In accuracy y_pred: ', y_pred)
        self.y_true_label = [int(y_val) for y_val in y]
        print ('In accuracy y true label: ', self.y_true_label)
        self.y_pred_label = y_pred.argmax(axis = 1)
        print ('In accuracy y pred label: ', self.y_pred_label)
        self.yoriginal.append(self.y_true_label)
        self.ypredicted.append(self.y_pred_label)
    #     print ('y_pred_label', y_pred_label, y_pred_label.shape)
        print ('current self.yoriginal: ', self.yoriginal)

        accuracy = accuracy_score(self.y_true_label, self.y_pred_label)
        print ('current accuracy: ', accuracy)
        return accuracy

    def precision_macro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the precision_macro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            accuracy 
        '''
        print ('In precision_macro y true label: ', self.y_true_label)
        print ('In precision_macro y pred label: ', self.y_pred_label)

        precision_macro = precision_score(self.y_true_label, self.y_pred_label, average = 'macro')
        print ('current precision_macro: ', precision_macro)
        return precision_macro
    
    
    def precision_micro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the precision micro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            accuracy 
        '''
        print ('In precision_micro y true label: ', self.y_true_label)
        print ('In precision_micro y pred label: ', self.y_pred_label)

        precision_micro = precision_score(self.y_true_label, self.y_pred_label, average = 'micro')
        print ('current precision_macro: ', precision_micro)
        return precision_micro
        
        
# def models(X, Y, model_name = 'random_forest', framework = 'W', results_path = '..\\MLresults\\', save_results = True):
#     '''
#     Arguments:
#     X, Y, PID groups so that strides of each person are either in training or in testing set
#     model: model_name, framework we wish to run the code for
#     Returns: predicted probabilities and labels for each class, stride and subject based evaluation metrics 
#     '''
#     Y_ = Y['label'] #Dropping the PID
#     groups_ = Y['PID']
#     #We use stratified group K-fold to sample our strides data
#     gkf = StratifiedGroupKFold(n_splits=5) 
#     scores={'accuracy': make_scorer(acc), 'precision_macro':make_scorer(precision_score, average = 'macro'), \
#             'precision_micro':make_scorer(precision_score, average = 'micro'), 'precision_weighted':make_scorer(precision_score, average = 'weighted'), 'recall_macro':make_scorer(recall_score, average = 'macro'), 'recall_micro':make_scorer(recall_score, average = 'micro'), 'recall_weighted':make_scorer(recall_score, average = 'weighted'), 'f1_macro': make_scorer(f1_score, average = 'macro'), 'f1_micro': make_scorer(f1_score, average = 'micro'), 'f1_weighted': make_scorer(f1_score, average = 'weighted'), 'auc_macro': make_scorer(roc_auc_score, average = 'macro', multi_class = 'ovo', needs_proba= True), 'auc_weighted': make_scorer(roc_auc_score, average = 'weighted', multi_class = 'ovo', needs_proba= True)}
#     if(model_name == 'random_forest'): #Random Forest
#         grid = {
#        'randomforestclassifier__n_estimators': [40,45,50],\
#        'randomforestclassifier__max_depth' : [15,20,25,None],\
#        'randomforestclassifier__class_weight': [None, 'balanced'],\
#        'randomforestclassifier__max_features': ['auto','sqrt','log2', None],\
#        'randomforestclassifier__min_samples_leaf':[1,2,0.1,0.05]
#         }
#         #For z-score scaling on training and use calculated coefficients on test set
#         rf_grid = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
#         grid_search = GridSearchCV(rf_grid, param_grid=grid, scoring=scores\
#                            , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
        
#     grid_search.fit(X, Y_, groups=groups_) #Fitting on the training set to find the optimal hyperparameters 
#     test_subjects_true_predicted_labels, stride_person_metrics = evaluate(grid_search, Y, yoriginal, ypredicted, framework, model_name, results_path, save_results)
#     return test_subjects_true_predicted_labels, stride_person_metrics

    

    def train(self, n_splits_ = 5):
        '''
        Tunes and trains skorch model for task generalization
        Arguments:
            model: Skorch model
            fullTrainLabelsList: List of training data labels and PIDs
            trainStridesList_norm: Normlized list of training sequences
            params: List of hyperparameters to optimize across
        Returns:
            Trained and tuned grid search object
        '''
        #shuffle data first
        self.X_sl, self.Y_sl, self.PID_sl = shuffle(self.X_sl, self.Y_sl, self.PID_sl, random_state = 0) 
        
        self.X_sl_ = [x for x in self.X_sl]
#         print ('X format: ', self.X_sl_train_[0])
        self.Y_sl_ = [int(y) for y in self.Y_sl]
        self.PID_sl_ = [int(pid) for pid in self.PID_sl]
        
#         print ('Shuffled PID and Y:', self.PID_sl_, self.Y_sl_)        
        self.yoriginal, self.ypredicted = [], []
        self.pipe = Pipeline([('scale', custom_StandardScaler()), ('net', self.model)])
#         self.pipe = Pipeline([('net', self.model)])
        
        gkf = StratifiedGroupKFold(n_splits=n_splits_)  
        scores = {'accuracy': self.accuracy_score_multi_class_cv, 'precision_macro': self.precision_macro_score_multi_class_cv, 'precision_micro': self.precision_micro_score_multi_class_cv}
#         , 'precision_weighted': self.evaluation_scores_multi_class['precision_weighted'], 'recall_macro': self.evaluation_scores_multi_class['recall_macro'], 'recall_micro': self.evaluation_scores_multi_class['recall_micro'], 'recall_weighted': self.evaluation_scores_multi_class['recall_weighted'], 'f1_macro': self.evaluation_scores_multi_class['f1_macro'], 'f1_micro': self.evaluation_scores_multi_class['f1_micro'], 'f1_weighted': self.evaluation_scores_multi_class['f1_weighted'], 'auc_macro': self.evaluation_scores_multi_class['auc_macro'], 'auc_weighted': self.evaluation_scores_multi_class['auc_weighted']}
        
        self.grid_search = GridSearchCV(self.pipe, param_grid = self.hyperparameter_grid, scoring = scores, \
                                        n_jobs = 1, cv = gkf.split(self.X_sl_, self.Y_sl_, groups=self.PID_sl_), \
                                        refit = False)
        print("grid search", self.grid_search)

#         print("Cross val split PIDs:\n")
#         for idx, (train, test) in enumerate(gkf.split(self.X_sl_train_, self.Y_sl_train_, groups=self.PID_sl_train_)):
#             print ('\nFold: ', idx+1)
#             print ('\nIndices in train: ', train, '\nIndices in test: ', test)
#             print('\nPIDs in TRAIN: ', np.unique(self.PID_sl_train[train], axis=0), '\nPIDs in TEST: ', \
#                   np.unique(self.PID_sl_train[test], axis=0))
#             print ('**************************************************************')

        #Skorch callback history to get loss to plot
        start_time = time.time()
        self.grid_search.fit(self.X_sl, self.Y_sl, groups=self.PID_sl)
        end_time = time.time()
        self.training_time = end_time - start_time
        print("\nTraining/ Cross validation time: ", self.training_time)
        
        
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
            
        print ('PIDs getting used in this run: ', self.trial['PID'].unique())        
        #Get dataloader 
        #Subject generalization W or WT framework 
        #Here the strides are normalized using within stride normalization, but frame counts are yet to normalized using training folds
        self.data = GaitDataset(self.data_path, self.labels_file, self.trial['PID'].unique(), framework = self.scenario)      
        self.create_folder_for_results()   
    
        if self.parameter_dict['behavior'] == 'train':
            self.model = self.create_model(model_, device_)
            self.X_sl = SliceDataset(self.data, idx = 0)
            self.Y_sl = SliceDataset(self.data, idx = 1)
            self.PID_sl = SliceDataset(self.data, idx = 2)
            self.train(n_splits_)
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


#To do/check:
#1. Frame count normalization as part of the pipeline, before it is fed to the model. Make sure that batch size parameter does not hinder this, because when finding out the normalizing mean and std, we need to use the full training data in n-1 folds and not just the batch size amount of samples. This will use creating our own standard scalar that is applicable only to frame count of x_data and not to body_coords.
#2. Training-validation curves for cv setup
#3. Count of parameters for selected best model in cv setup
#4. Do we need evaluate behaviour for this cv setup case
#5. Do we need resume training behaviour for this cv setup case
#6. How can we save models/load trained models in this cv setup case
#7. Retain indices in y_true in acc() to do groupby(pid) using corresponding indices later for subject-wise metrics 