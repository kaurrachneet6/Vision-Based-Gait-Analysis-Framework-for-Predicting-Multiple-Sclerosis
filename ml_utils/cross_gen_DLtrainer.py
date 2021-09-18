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
from ml_utils.DLutils import save_model, load_model, design, custom_StandardScaler, MyCheckpoint, FixRandomSeed

class GaitTrainer():
    def __init__(self, parameter_dict, hyperparameter_grid, config_path):
        self.parameter_dict = parameter_dict
        self.data_path = self.parameter_dict['path'] + self.parameter_dict['data_path']
        self.labels_file = self.parameter_dict['path'] + self.parameter_dict['labels_file']    
        self.labels = pd.read_csv(self.labels_file, index_col = 0)
        self.framework = self.parameter_dict['framework'] #Defining the framework of interest #'task_and_subject_WtoWT'
        self.train_framework = self.parameter_dict['train_framework']
        self.test_framework = self.parameter_dict['test_framework']
        self.hyperparameter_grid = hyperparameter_grid
        self.save_results_path = self.parameter_dict['results_path']  + self.framework + '/'+ self.parameter_dict['model_path']
        self.save_results_prefix = self.parameter_dict['prefix_name'] + '_'
        self.save_results = self.parameter_dict['save_results']
        self.config_path = config_path
        
        
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
        

    def compute_train_test_indices_split(self, n_splits_ = 5):
        '''
        For task+subject generalization framework, since our train and test indices are custom made, i.e. train indices are from
        trial W and test indices are from trial WT. Further, since 7 subjects only exist in trial W, we include them only in 
        training indices and 1 subject only exists in trial WT, we include them only in testing indices. For other 25 common
        subjects we do stratified group five fold cross validation based on subject PIDs and thus no same subject can have strides 
        both in training and in validation sets. We add the strides from 7 only training subjects to strides assigned via CV to training
        and similarly add the strides from 1 only testing subject to strides assigned via CV to testing. Thus the training only and 
        testing only strides remain common in all folds. 
        Arguments:
            train_test_concatenated: the concatendated dataframe with both training and testing strides appended 
            X_train_common: X (including 91 features) for training set with common PIDs only 
            Y_train_common: Y (labels) for training set with common PIDs only 
            train_pids: PIDs for subjects that only exist in training 
            test_pids: PIDs for subjects that only exist in testing 
            train_framework: training task 
            test_framework: testing task 
        Returns:
            train_indices: list of lists of .iloc indices for training folds 
            test_indices: list of lists of .iloc indices for testing folds 
        '''
        first_index = True
        
        #List to append lists of training and test indices for each CV fold
        self.train_indices, self.test_indices = [], []
        #PIDs define the groups for stratified group 5-fold CV
        groups_ = self.Y_train_common['PID']

        #We use stratified group K-fold to sample our strides data
        gkf = StratifiedGroupKFold(n_splits=n_splits_) 

        #Indices for strides of subjects that exist only in training set 
        train_only_indices = self.train_test_concatenated[(self.train_test_concatenated.PID.isin(self.train_pids)) \
                                                     & (self.train_test_concatenated.scenario==self.train_framework)].index
        #Indices for strides of subjects that exist only in testing set 
        test_only_indices = self.train_test_concatenated[(self.train_test_concatenated.PID.isin(self.test_pids)) \
                                                     & (self.train_test_concatenated.scenario==self.test_framework)].index

        #Computing the CV fold indices for common subjects in training and testing 
        for idx, (train_idx, test_idx) in enumerate(gkf.split(self.X_train_common, self.Y_train_common['label'], groups=groups_)):
#             print ('In fold: ', idx)
            #PIDs for train indices using Stratified group 5-fold split
            train_split_pids = groups_.iloc[train_idx].unique()
#             print ('train_pids', train_split_pids)
            #Indices for training using CV split in each fold 
            train_split_indices = self.train_test_concatenated[(self.train_test_concatenated.PID.isin(train_split_pids)) \
                                                         & (self.train_test_concatenated.scenario==self.train_framework)].index
            #Concatenating the indices of strides for PIDs in training only and CV split training PIDs 
            train_split_indices = train_split_indices.union(train_only_indices)
        #     print (train_split_indices, train_split_indices.shape)
            #Appending the training indices for the current fold 
            self.train_indices.append(train_split_indices)

            #PIDs for test indices using Stratified group 5-fold split
            test_split_pids = groups_.iloc[test_idx].unique()
#             print ('test_pids', test_split_pids)
            #Indices for testing using CV split in each fold 
            test_split_indices = self.train_test_concatenated[(self.train_test_concatenated.PID.isin(test_split_pids)) \
                                                         & (self.train_test_concatenated.scenario==self.test_framework)].index
            #Concatenating the indices of strides for PIDs in testing only and CV split testing PIDs 
            if first_index:
                #Concatenating the indices of strides for PIDs in testing only and CV split testing PIDs 
                test_split_indices = test_split_indices.union(test_only_indices)
                first_index = False
        #     print (test_split_indices, test_split_indices.shape)
            #Appending the testing indices for the current fold 
            self.test_indices.append(test_split_indices)

        #Computing the .iloc indices from the .loc indices of the training and testing strides 
        self.train_indices = [self.train_test_concatenated.reset_index().index[self.train_test_concatenated.index.isin(self.train_indices[i])] for i in range(len(self.train_indices))]
        self.test_indices = [self.train_test_concatenated.reset_index().index[self.train_test_concatenated.index.isin(self.test_indices[i])] for i in range(len(self.test_indices))]


    def create_folder_for_results(self):
        #Create folder for saving results
        time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
        self.save_path = self.save_results_path + self.save_results_prefix + time_now+"/"
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
            max_epochs = 2000,
            lr = .0001,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            device = device_,
            iterator_train__shuffle=True,
            train_split = skorch.dataset.CVSplit(5, random_state = 0), 
            batch_size= -1, #Batch size = -1 means full data at once 
            callbacks=[EarlyStopping(patience = 100, lower_is_better = True, threshold=0.0001), 
            (FixRandomSeed()), 
            #('lr_scheduler', LRScheduler(policy=torch.optim.lr_scheduler.StepLR, step_size = 500)),
            (EpochScoring(scoring=DLutils.accuracy_score_multi_class, lower_is_better = False, on_train = True, name = "train_acc")),
            (EpochScoring(scoring=DLutils.accuracy_score_multi_class, lower_is_better = False, on_train = False, name = "valid_acc")),
            (MyCheckpoint(f_params='params.pt', f_optimizer='optimizer.pt', f_criterion='criterion.pt', f_history='history.json', f_pickle=None, fn_prefix='train_end_', dirname= self.save_path)),
#             (PrintLog(keys_ignored=None))
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
        self.y_pred = net.predict(X)
#         print ('In accuracy y_pred: ', y_pred)
        self.y_true_label = y
        self.y_pred_label = self.y_pred.argmax(axis = 1)
#         print ('In accuracy y pred label: ', self.y_pred_label)
        self.yoriginal.append(self.y_true_label)
        self.ypredicted.append(self.y_pred_label)
        accuracy = accuracy_score(self.y_true_label, self.y_pred_label)
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
            precision macro
        '''
        precision_macro = precision_score(self.y_true_label, self.y_pred_label, average = 'macro')
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
            precision micro
        '''
        precision_micro = precision_score(self.y_true_label, self.y_pred_label, average = 'micro')
        return precision_micro
        
    def precision_weighted_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the precision weighted using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            precision weighted
        '''
        precision_weighted = precision_score(self.y_true_label, self.y_pred_label, average = 'weighted')
        return precision_weighted
 
    def recall_macro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the recall macro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            recall macro
        '''
        recall_macro = recall_score(self.y_true_label, self.y_pred_label, average = 'macro')
        return recall_macro
    
    
    def recall_micro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the recall micro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            recall micro
        '''
        recall_micro = recall_score(self.y_true_label, self.y_pred_label, average = 'micro')
        return recall_micro
        
    def recall_weighted_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the recall weighted using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            recall weighted
        '''
        recall_weighted = recall_score(self.y_true_label, self.y_pred_label, average = 'weighted')
        return recall_weighted
   
    def f1_macro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the f1 macro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            f1 macro
        '''
        f1_macro = f1_score(self.y_true_label, self.y_pred_label, average = 'macro')
        return f1_macro
    
    
    def f1_micro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the f1 micro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            f1 micro
        '''
        f1_micro = f1_score(self.y_true_label, self.y_pred_label, average = 'micro')
        return f1_micro
        
    def f1_weighted_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the f1 weighted using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            f1 weighted
        '''
        f1_weighted = f1_score(self.y_true_label, self.y_pred_label, average = 'weighted')
        return f1_weighted
    
    def auc_macro_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the auc macro using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            auc macro
        '''
        auc_macro = roc_auc_score(self.y_true_label, self.y_pred, average = 'macro', multi_class = 'ovo')
        return auc_macro
    

    def auc_weighted_score_multi_class_cv(self, net, X, y):
        '''
        Function to compute the auc weighted using the softmax probabilities predicted via skorch neural net in a 
        cross validation type setting 
        Arguments:
            net: skorch model
            X: data 
            y: true target labels
        Returns:
            auc weighted
        '''
        auc_weighted = roc_auc_score(self.y_true_label, self.y_pred, average = 'weighted', multi_class = 'ovo')
        return auc_weighted
    
    
    def evaluate(self, n_splits_ = 5):
        '''
        Arguments: 
            trained model, test set, true and predicted labels for test set, framework and model name 
            save_results: Whether to save the results or not 
        Returns: 
            predicted probabilities and labels for each class, stride and subject based evaluation metrics 
        Saves the csv files for stride wise predictions and subject wise predictions for confusion matrix 
        '''
        #For creating the stride wise confusion matrix, we append the true and predicted labels for strides in each fold to this 
        #test_strides_true_predicted_labels dataframe 
        test_strides_true_predicted_labels = pd.DataFrame()
        #For creating the subject wise confusion matrix, we append the true and predicted labels for subjects in each fold to this
        #test_subjects_true_predicted_labels dataframe
        self.test_subjects_true_predicted_labels = pd.DataFrame()

        best_index = self.grid_search.cv_results_['mean_test_accuracy'].argmax()
        self.best_params = self.grid_search.cv_results_['params'][best_index]
        print('best_params: ', self.best_params)

        if self.save_results:
            best_params_file = open(self.save_path + "best_parameters.txt","w") 
            best_params_file.write(str(self.best_params))
            best_params_file.close() 
            
        #Count of parameters in the selected model
#         print ('Module: ', self.pipe.set_params(**self.best_params)['net'].module)
        self.total_parameters = sum(p.numel() for p in self.pipe.set_params(**self.best_params)['net'].module.parameters())     
        self.trainable_params =  sum(p.numel() for p in self.pipe.set_params(**self.best_params)['net'].module.parameters() if p.requires_grad)
        self.nontrainable_params = self.total_parameters - self.trainable_params
        trueY_df = pd.DataFrame(data = np.array((self.PID_sl, self.Y_sl_)).T, columns = ['PID', 'label'])
        person_acc, person_p_macro, person_p_micro, person_p_weighted, person_p_class_wise, person_r_macro, person_r_micro, person_r_weighted, person_r_class_wise, person_f1_macro, person_f1_micro, person_f1_weighted, person_f1_class_wise, person_auc_macro, person_auc_weighted = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        class_wise_scores = {'precision_class_wise': [], 'recall_class_wise': [], 'f1_class_wise': []}

        for i in range(n_splits_):
            #For each fold, there are 2 splits: test and train (in order) and we need to retrieve the index 
            #of only test set for required 5 folds (best index)
            temp = trueY_df.loc[self.yoriginal[(best_index*n_splits_) + (i)].index] #True labels for the test strides in each fold
            temp['pred'] = self.ypredicted[(best_index*n_splits_) + (i)] #Predicted labels for the strides in the test set in each fold
    #         display('temp', temp)
    #         print ('temp_pred', temp['pred'])
            #Appending the test strides' true and predicted label for each fold to compute stride-wise confusion matrix 
            test_strides_true_predicted_labels = test_strides_true_predicted_labels.append(temp)

            x = temp.groupby('PID')['pred'].value_counts().unstack()
    #         print ('x', x)
            #Input for subject wise AUC is probabilities at columns [0, 1, 2]
            proportion_strides_correct = pd.DataFrame(columns = [0, 1, 2])
            probs_stride_wise = x.divide(x.sum(axis = 1), axis = 0).fillna(0)
            proportion_strides_correct[probs_stride_wise.columns] = probs_stride_wise
            proportion_strides_correct.fillna(0, inplace=True)
            proportion_strides_correct['True Label'] = trueY_df.groupby('PID').first()
            #Input for precision, recall and F1 score
            proportion_strides_correct['Predicted Label'] = proportion_strides_correct[[0, 1, 2]].idxmax(axis = 1) 
            #Appending the test subjects' true and predicted label for each fold to compute subject-wise confusion matrix 
            self.test_subjects_true_predicted_labels = self.test_subjects_true_predicted_labels.append(proportion_strides_correct)          

            #Class-wise metrics for stride evaluation metrics 
            class_wise_scores['precision_class_wise'].append(precision_score(temp['label'], temp['pred'], average = None))
            class_wise_scores['recall_class_wise'].append(recall_score(temp['label'], temp['pred'], average = None))
            class_wise_scores['f1_class_wise'].append(f1_score(temp['label'], temp['pred'], average = None))

            #Person wise metrics for each fold 
            person_acc.append(accuracy_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label']))
            person_p_macro.append(precision_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                           average = 'macro'))
            person_p_micro.append(precision_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                           average = 'micro'))
            person_p_weighted.append(precision_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                           average = 'weighted'))
            person_p_class_wise.append(precision_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                           average = None))

            person_r_macro.append(recall_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                        average = 'macro'))
            person_r_micro.append(recall_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                        average = 'micro'))
            person_r_weighted.append(recall_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                        average = 'weighted'))
            person_r_class_wise.append(recall_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                        average = None))

            person_f1_macro.append(f1_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                      average = 'macro'))
            person_f1_micro.append(f1_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                      average = 'micro'))
            person_f1_weighted.append(f1_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                      average = 'weighted'))
            person_f1_class_wise.append(f1_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label'], \
                                      average = None))

            person_auc_macro.append(roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct[[0, 1, 2]], \
                                            multi_class = 'ovo', average= 'macro'))
            person_auc_weighted.append(roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct[[0, 1, 2]], \
                                            multi_class = 'ovo', average= 'weighted'))

        #Mean and standard deviation for person-based metrics 
        person_means = [np.mean(person_acc), np.mean(person_p_macro), np.mean(person_p_micro), np.mean(person_p_weighted), list(map(mean, zip(*person_p_class_wise))), np.mean(person_r_macro), np.mean(person_r_micro), np.mean(person_r_weighted), list(map(mean, zip(*person_r_class_wise))), np.mean(person_f1_macro), np.mean(person_f1_micro), np.mean(person_f1_weighted), list(map(mean, zip(*person_p_class_wise))), np.mean(person_auc_macro), np.mean(person_auc_weighted)]

        person_stds = [np.std(person_acc), np.std(person_p_macro), np.std(person_p_micro), np.std(person_p_weighted), list(map(stdev, zip(*person_p_class_wise))), np.std(person_r_macro), np.std(person_r_micro), np.std(person_r_weighted), list(map(stdev, zip(*person_r_class_wise))), np.std(person_f1_macro), np.std(person_f1_micro), np.std(person_f1_weighted), list(map(stdev, zip(*person_p_class_wise))), np.std(person_auc_macro), np.std(person_auc_weighted)]

            #Stride-wise metrics 
        stride_metrics_mean, stride_metrics_std = [], [] #Mean and SD of stride based metrics - Acc, P, R, F1, AUC (in order)
        scores = {'accuracy': self.accuracy_score_multi_class_cv, 'precision_macro': self.precision_macro_score_multi_class_cv, 'precision_micro': self.precision_micro_score_multi_class_cv, 'precision_weighted': self.precision_weighted_score_multi_class_cv, 'precision_class_wise':[], 'recall_macro': self.recall_macro_score_multi_class_cv, 'recall_micro': self.recall_micro_score_multi_class_cv, 'recall_weighted': self.recall_weighted_score_multi_class_cv, 'recall_class_wise': [], 'f1_macro': self.f1_macro_score_multi_class_cv, 'f1_micro': self.f1_micro_score_multi_class_cv, 'f1_weighted': self.f1_weighted_score_multi_class_cv, 'f1_class_wise': [], 'auc_macro': self.auc_macro_score_multi_class_cv, 'auc_weighted': self.auc_weighted_score_multi_class_cv}

        for score in scores:
            try:
                stride_metrics_mean.append(self.grid_search.cv_results_['mean_test_'+score][best_index])
                stride_metrics_std.append(self.grid_search.cv_results_['std_test_'+score][best_index])
            except:
                stride_metrics_mean.append(list(map(mean, zip(*class_wise_scores[score]))))
                stride_metrics_std.append(list(map(stdev, zip(*class_wise_scores[score]))))
        print('\nStride-based model performance (mean): ', stride_metrics_mean)
        print('\nStride-based model performance (standard deviation): ', stride_metrics_std)

        print('\nPerson-based model performance (mean): ', person_means)
        print('\nPerson-based model performance (standard deviation): ', person_stds)

        #Saving the stride and person wise true and predicted labels for calculating the 
        #stride and subject wise confusion matrix for each model
        if self.save_results:
            test_strides_true_predicted_labels.to_csv(self.save_path  + 'stride_wise_predictions_' + self.framework + '.csv')
            self.test_subjects_true_predicted_labels.to_csv(self.save_path + 'person_wise_predictions_' + self.framework + '.csv')

        #Plotting and saving the sequence and subject wise confusion matrices 
        #Sequence wise confusion matrix
        plt.figure()
        confusion_matrix = pd.crosstab(test_strides_true_predicted_labels['label'], test_strides_true_predicted_labels['pred'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
        if self.save_results:
            plt.savefig(self.save_path  + 'CFmatrix_cross_generalize_' + self.framework + '_stride_wise.png', dpi = 350)
        plt.show();

        #Plotting and saving the subject wise confusion matrix 
        plt.figure()
        confusion_matrix = pd.crosstab(self.test_subjects_true_predicted_labels['True Label'], self.test_subjects_true_predicted_labels['Predicted Label'], \
                                       rownames=['Actual'], colnames=['Predicted'], margins = True)
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
        if self.save_results:
            plt.savefig(self.save_path  +'CFmatrix_cross_generalize_' + self.framework + '.png', dpi = 350)
        plt.show(); 
        
        
        stride_person_metrics = [stride_metrics_mean, stride_metrics_std, person_means, person_stds, [self.training_time], [self.best_params], [self.total_parameters], [self.trainable_params], [self.nontrainable_params], [str(self.total_epochs)]]
        
        self.metrics = pd.DataFrame(columns = [self.save_results_prefix]) #Dataframe to store accuracies for each ML model for raw data 
        self.metrics[self.save_results_prefix] = sum(stride_person_metrics, [])
        stride_scoring_metrics = ['stride_accuracy', 'stride_precision_macro', 'stride_precision_micro', 'stride_precision_weighted', \
                 'stride_precision_class_wise', 'stride_recall_macro', 'stride_recall_micro', \
                 'stride_recall_weighted', 'stride_recall_class_wise', \
                 'stride_F1_macro', 'stride_F1_micro', 'stride_F1_weighted', 'stride_F1_class_wise', \
                 'stride_AUC_macro', 'stride_AUC_weighted']
        person_scoring_metrics = ['person_accuracy', 'person_precision_macro', 'person_precision_micro', \
                 'person_precision_weighted', 'person_precision_class_wise', 'person_recall_macro', 'person_recall_micro', \
                 'person_recall_weighted', 'person_recall_class_wise', \
                 'person_F1_macro', 'person_F1_micro', 'person_F1_weighted', 'person_F1_class_wise', \
                 'person_AUC_macro', 'person_AUC_weighted']   
        print ('Metrics', self.metrics)
#         print ('Metrics Index', [i + '_mean' for i in stride_scoring_metrics] + [i + '_std' for i in stride_scoring_metrics] + [i + '_mean' for i in person_scoring_metrics] + [i + '_std' for i in person_scoring_metrics] + ['training_time', 'best_parameters']  + ['total_parameters', 'trainable_params', 'nontrainable_params', 'Total Epochs'])
        self.metrics.index = [i + '_mean' for i in stride_scoring_metrics] + [i + '_std' for i in stride_scoring_metrics] + [i + '_mean' for i in person_scoring_metrics] + [i + '_std' for i in person_scoring_metrics] + ['training_time', 'best_parameters']  + ['total_parameters', 'trainable_params', 'nontrainable_params', 'Total Epochs']
    
        #Saving the evaluation metrics and tprs/fprs/rauc for the ROC curves 
        if self.save_results:
            self.metrics.to_csv(self.save_path + 'cross_generalize_'+self.framework+'_result_metrics.csv')

            
    #ROC curves 
    def plot_ROC(self):
        '''
        Function to plot the ROC curve and confusion matrix for model given in ml_model name 
        Input: ml_models (name of models to plot the ROC for),  test_Y (true test set labels with PID), 
            predicted_probs_person (predicted test set probabilities for all 3 classes - HOA/MS/PD), framework (WtoWT / VBWtoVBWT)
        Plots and saves the ROC curve with individual class-wise plots and micro/macro average plots 
        '''
        n_classes = 3 #HOA/MS/PD
        cohort = ['HOA', 'MS', 'PD']
        #Binarizing/getting dummies for the true labels i.e. class 1 is represented as 0, 1, 0
        test_features_binarize = pd.get_dummies(self.test_subjects_true_predicted_labels['True Label'].values)     
        sns.despine(offset=0)
        linestyles = ['-', '-', '-', '-.', '--', '-', '--', '-', '--']
        colors = ['b', 'magenta', 'cyan', 'g',  'red', 'violet', 'lime', 'grey', 'pink']

        fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(6, 4.5))
        axes.plot([0, 1], [0, 1], linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
        # person-based prediction probabilities for class 0: HOA, 1: MS, 2: PD

        # Compute ROC curve and ROC area for each class
        tpr, fpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes): #n_classes = 3
            fpr[i], tpr[i], _ = roc_curve(test_features_binarize.iloc[:, i], self.test_subjects_true_predicted_labels.loc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            #Plotting the ROCs for the three classes separately
            axes.plot(fpr[i], tpr[i], label = cohort[i] +' ROC (AUC = '+ str(round(roc_auc[i], 3))
                +')', linewidth = 3, alpha = 0.8, linestyle = linestyles[i], color = colors[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_features_binarize.values.ravel(),\
                                                  self.test_subjects_true_predicted_labels[[0, 1, 2]].values.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #Plotting the micro average ROC 
        axes.plot(fpr["micro"], tpr["micro"], label= 'micro average ROC (AUC = '+ str(round(roc_auc["micro"], 3))
                +')', linewidth = 3, alpha = 0.8, linestyle = linestyles[3], color = colors[3])

        #Compute the macro-average ROC curve and AUC value
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)])) # First aggregate all false positive rates
        mean_tpr = np.zeros_like(all_fpr) # Then interpolate all ROC curves at this points
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes  # Finally average it and compute AUC
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        #Macro average AUC of ROC value 
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])    
        #Plotting the macro average AUC
        axes.plot(fpr["macro"], tpr["macro"], label= 'macro average ROC (AUC = '+ str(round(roc_auc["macro"], 3))
            +')', linewidth = 3, alpha = 0.8, linestyle = linestyles[4], color = colors[4])

        axes.set_ylabel('True Positive Rate')
        axes.set_title('Cross generalization '+ self.framework)
        plt.legend()
        # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

        axes.set_xlabel('False Positive Rate')
        plt.tight_layout()
        if self.save_results:
            plt.savefig(self.save_path+ 'ROC_cross_generalize_' + self.framework +  '_'+ self.save_results_prefix  + '.png', dpi = 350)
        plt.show()
    
    
    
    #Main training paradigm 
    def train(self, n_splits_ = 5, feature_importance = False):
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
        self.X_sl_ = [x for x in self.X_sl]
#         print ('X format: ', self.X_sl_train_[0])
        self.Y_sl_ = [int(y) for y in self.Y_sl]
#         print ('In train, Y unique', np.unique(self.Y_sl_), 'length = ', len(self.Y_sl_))
        self.PID_sl_ = [int(pid) for pid in self.PID_sl]
        
#         print ('Shuffled PID and Y:', self.PID_sl_, self.Y_sl_)        
        self.yoriginal, self.ypredicted = [], []
        if not feature_importance:
            self.pipe = Pipeline([('scale', custom_StandardScaler()), ('net', self.model)])
        else:
            self.pipe = Pipeline([('scale', custom_StandardScaler()), ('permutation', PermuteTransform(self.feature_indices)), ('net', self.model)])
        self.scores = {'accuracy': self.accuracy_score_multi_class_cv, 'precision_macro': self.precision_macro_score_multi_class_cv, 'precision_micro': self.precision_micro_score_multi_class_cv, 'precision_weighted': self.precision_weighted_score_multi_class_cv, 'recall_macro': self.recall_macro_score_multi_class_cv, 'recall_micro': self.recall_micro_score_multi_class_cv, 'recall_weighted': self.recall_weighted_score_multi_class_cv, 'f1_macro': self.f1_macro_score_multi_class_cv, 'f1_micro': self.f1_micro_score_multi_class_cv, 'f1_weighted': self.f1_weighted_score_multi_class_cv, 'auc_macro': self.auc_macro_score_multi_class_cv, 'auc_weighted': self.auc_weighted_score_multi_class_cv}
        
        self.grid_search = GridSearchCV(self.pipe, param_grid = self.hyperparameter_grid, scoring = self.scores\
                           , n_jobs = 1, cv = zip(self.train_indices, self.test_indices), refit=False)
        print("grid search", self.grid_search)

#         print("Cross val split PIDs:\n")
#         for idx, (train, test) in enumerate(zip(self.train_indices, self.test_indices)):
#             print ('\nFold: ', idx+1)
#             print ('\nIndices in train: ', train, '\nIndices in test: ', test)
#             print('\nPIDs in TRAIN: ', np.unique(self.PID_sl[train], axis=0), '\nPIDs in TEST: ', \
#                   np.unique(self.PID_sl[test], axis=0))
#             print('\nY in TRAIN: ', pd.Series(self.Y_sl)[train], '\nY in TEST: ', \
#                   pd.Series(self.Y_sl)[test])          
#             print ('**************************************************************')

        #Skorch callback history to get loss to plot
        start_time = time.time()
        self.grid_search.fit(self.X_sl, pd.Series(self.Y_sl))
        end_time = time.time()
        self.training_time = end_time - start_time
        print("\nTraining/ Cross validation time: ", self.training_time)
        
   

    #Learning curves     
    def learning_curves(self):
        '''
        To plot the training/validation loss and accuracy (stride-wise) curves over epochs across the n_splits folds 
        '''
        best_index = self.grid_search.cv_results_['mean_test_accuracy'].argmax()
        self.best_params = self.grid_search.cv_results_['params'][best_index]
        pipe_optimized = self.pipe.set_params(**self.best_params)
        #List of history dataframes over n_splits folds 
        histories = []  
        self.total_epochs = []   
        
        for fold, (train_ix, val_ix) in enumerate(zip(self.train_indices, self.test_indices)):
            # select rows for train and test
            trainX, trainY, valX, valY = self.X_sl[train_ix], self.Y_sl[train_ix], self.X_sl[val_ix], self.Y_sl[val_ix]
            # fit model
            pipe_optimized.fit(trainX, trainY)
            history_fold = pd.DataFrame(pipe_optimized['net'].history)
#             print ('History for fold', fold+1, '\n')
#             display(history_fold)
            if self.save_results:
                history_fold.to_csv(self.save_path + 'history_fold_' + str(fold+1) + '.csv')
            histories.append(history_fold)
          
        for idx in range(len(histories)):
            model_history = histories[idx]
            epochs = model_history['epoch'].values #start from 1 instead of zero
            self.total_epochs.append(len(epochs))
            train_loss = model_history['train_loss'].values
    #         print (train_loss)
            valid_loss = model_history['valid_loss'].values
            train_acc = model_history['train_acc'].values
            valid_acc = model_history['valid_acc'].values
            #print("epochs", epochs, len(epochs))
            #print("train_acc", train_acc, len(train_acc))
            #print("train_loss", train_loss, len(train_loss))
            #print("valid_loss", valid_loss, len(valid_loss))
            plt.plot(epochs,train_loss,'g--'); #Dont print the last one for 3 built in
            plt.plot(epochs,valid_loss,'r-');
            try:
                plt.plot(epochs,train_acc,'b--');
            except:
                plt.plot(epochs,train_acc[:-1],'b-');
            #plt.plot(np.arange(len(train_acc)),train_acc, 'b-'); #epochs and train_acc are off by one
            try:
                plt.plot(epochs,valid_acc, 'm-');
            except:
                plt.plot(epochs[:-1], valid_acc, 'm-');
        plt.title('Training/Validation loss and accuracy Curves');
        plt.xlabel('Epochs');
        plt.ylabel('Cross entropy loss/Accuracy');
        plt.legend(['Train loss','Validation loss', 'Train Accuracy', 'Validation Accuracy']); 
        if self.save_results:
            plt.savefig(self.save_path + 'learning_curve', dpi = 350)
        plt.show()
        plt.close()
        
        
        
    #Main setup
    def cross_gen_setup(self, model_ = None, device_ = torch.device("cuda"), n_splits_ = 5, datastream = "All"):
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

        self.X_train_common = self.train_trial_train_commonPID.drop(['PID', 'label'], axis = 1)
        self.Y_train_common = self.train_trial_train_commonPID[['PID', 'label']]

        self.train_test_concatenated = self.labels[self.labels.scenario.isin([self.train_framework, self.test_framework])].reset_index().drop('index', axis = 1)    
        
        #Get dataloader 
        #Here the strides are normalized using within stride normalization, but frame counts are yet to normalized using training folds
        self.data = GaitDataset(self.data_path, self.labels_file, self.train_test_concatenated['PID'].unique(), framework = [self.train_framework, self.test_framework], datastream = datastream)
        #Computing the X (91 features), Y (PID, label) for the models 
        self.X_sl = SliceDataset(self.data, idx = 0)
        self.Y_sl = SliceDataset(self.data, idx = 1)
        self.PID_sl = SliceDataset(self.data, idx = 2)
        print ('Shapes', self.train_test_concatenated.shape, self.X_sl.shape, self.Y_sl.shape) #1176+1651
        
        #Shuffling the concatenated data
        self.train_test_concatenated, self.X_sl, self.Y_sl, self.PID_sl = shuffle(self.train_test_concatenated, self.X_sl, self.Y_sl, self.PID_sl, random_state = 0) 
        
        #Computing the training and test set indices for the CV folds         
        self.compute_train_test_indices_split(n_splits_)
        self.create_folder_for_results() 
        self.torch_model = model_
        if self.parameter_dict['behavior'] == 'train':
            self.model = self.create_model(self.torch_model, device_)            
            self.train(n_splits_)
            cv_results = pd.DataFrame(self.grid_search.cv_results_)
            if self.save_results:
                cv_results.to_csv(self.save_path+"cv_results.csv")
            self.learning_curves()    
        self.evaluate(n_splits_) 
        self.plot_ROC() 

    
    
    '''
    Permutation Feature Importance 
    '''
    def cross_gen_perm_imp_initial_setup(self, model_ = None, device_ = torch.device("cuda"), n_splits_ = 5):
        '''
        Permutation feature importance for cross generalization initial setup
        '''       
        self.extract_train_test_common_PIDs()
        design()
        
        #Trial W for training 
        self.trial_train = self.labels[self.labels['scenario']==self.train_framework] #Full trial W with all 32 subjects 
        #Trial WT for testing 
        self.trial_test = self.labels[self.labels['scenario']==self.test_framework] #Full trial WT with all 26 subjects 
        
        #Training only data with strides from W
        train_only_trial_train = self.trial_train[self.trial_train.PID.isin(self.train_pids)] #subset of trial W with subjects only present in trial W but not in trial WT
        #Testing only data with strides from WT
        test_only_trial_test = self.trial_test[self.trial_test.PID.isin(self.test_pids)] #subset of trial WT with subjects only present in trial WT but not in trial W

        #Training data with strides from W for common PIDs in trials W and WT
        self.train_trial_train_commonPID = self.trial_train[self.trial_train.PID.isin(self.common_pids)] #subset of trial W with common subjects in trial W and WT
        #Testing data with strides from WT for common PIDs in trials W and WT
        self.test_trial_test_commonPID = self.trial_test[self.trial_test.PID.isin(self.common_pids)] #subset of trial W with common subjects in trial W and WT

        self.X_train_common = self.train_trial_train_commonPID.drop(['PID', 'label'], axis = 1)
        self.Y_train_common = self.train_trial_train_commonPID[['PID', 'label']]
        self.train_test_concatenated = self.labels[self.labels.scenario.isin([self.train_framework, self.test_framework])].reset_index().drop('index', axis = 1)    
        
        #Get dataloader 
        #Here the strides are normalized using within stride normalization, but frame counts are yet to normalized using training folds
        self.data = GaitDataset(self.data_path, self.labels_file, self.train_test_concatenated['PID'].unique(), framework = [self.train_framework, self.test_framework], datastream = "All")
        #Computing the X (91 features), Y (PID, label) for the models 
        self.X_sl = SliceDataset(self.data, idx = 0)
        self.Y_sl = SliceDataset(self.data, idx = 1)
        self.PID_sl = SliceDataset(self.data, idx = 2)
        
        #Shuffling the concatenated data
        self.train_test_concatenated, self.X_sl, self.Y_sl, self.PID_sl = shuffle(self.train_test_concatenated, self.X_sl, self.Y_sl, self.PID_sl, random_state = 0) 
        
        #Computing the training and test set indices for the CV folds         
        self.compute_train_test_indices_split(n_splits_)
        self.create_folder_for_results() 
        self.torch_model = model_
        self.model = self.create_model(self.torch_model, device_) 
        self.save_results = False
        self.save_results_path = self.parameter_dict['results_path'] + '../PermImpResults/' + self.framework + '/' + self.parameter_dict['model_path']
        self.total_epochs = 0
        self.train(n_splits_) #Trainign only for 2 splits here because we just want the metrics index from this run and not the real results 
        self.evaluate(n_splits_) #To get self.metrics.index for making the dataframe of metrics for FI
        self.perm_imp_results_df = pd.DataFrame(index = self.metrics.index)
        
        
    def cross_gen_perm_imp_single_feature(self, feature, n_splits_):  
        '''
        Running the permutation feature importance for a single feature(group) say, left big toe
        Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html#fn35
        '''
        #Column indices to permute in the X_sl_test_original to generate a new X_sl_test to predict and evaluate the trained model on 
        self.feature_indices = self.data.__define_column_indices_FI__(feature)
        print (self.feature_indices)
        #Repeating the shuffling 5 times for randomness in permutations
        for idx in range(5):
            #Restoring back the original unpermuted version 
            self.X_sl = SliceDataset(self.data, idx = 0)
            #Shuffling the features of interest
            self.train(n_splits_, feature_importance = True)
            #Predicting the best trained model on shuffled data and computing the metrics 
            self.save_results_prefix = feature + '_' + str(idx)
            self.evaluate(n_splits_) 
            #Saving the metrics 
            self.perm_imp_results_df[self.save_results_prefix] = self.metrics
        feature_cols = [s for s in self.perm_imp_results_df.columns if feature in s]
        #Aggregating the mean and SD from the 5 random runs
        self.perm_imp_results_df[feature + '_' + 'mean'] = self.perm_imp_results_df[feature_cols].apply(pd.to_numeric, args=['coerce']).mean(axis=1, skipna=False)
        self.perm_imp_results_df[feature + '_' + 'std'] = self.perm_imp_results_df[feature_cols].apply(pd.to_numeric, args=['coerce']).std(axis=1, skipna=False)        
        
        
    def cross_gen_perm_imp_main_setup(self, model = None, device_ = torch.device("cuda"), n_splits_ = 5):
        '''
        Main setup for the permutation feature importance for cross gen 
        Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html#fn35
        '''
        self.cross_gen_perm_imp_initial_setup(model, device_, n_splits_)
        
        #12 Feature groups to explore the importance for 
        features = ['right hip', 'right knee', 'right ankle', 'left hip', 'left knee', 'left ankle', 'left toe 1', 'left toe 2', 'left heel', 'right toe 1', 'right toe 2', 'right heel']
        global runs 
        runs = 0
        
        for feature in features:
            #For all 12 feature groups 
            print ('Running for ', feature)
            self.cross_gen_perm_imp_single_feature(feature, n_splits_)
        
        display(self.perm_imp_results_df)
        #Saving all the 7*12 columns for all 12 feature groups and all 5 runs+2(mean/std)
        self.perm_imp_results_df.to_csv(self.save_results_path + 'Permutation_importance_all_results.csv')
        
        result_mean_cols = [s for s in  self.perm_imp_results_df.columns if 'mean' in s]
        result_std_cols = [s for s in  self.perm_imp_results_df.columns if 'std' in s]
        main_result_cols = result_mean_cols + result_std_cols
        #Saving only the mean and std per 12 feature groups (24 columns) that will be used to plot FI later
        self.perm_imp_results_df[main_result_cols].to_csv(self.save_results_path + 'Permutation_importance_only_main_results.csv')
        
     
    ''' SHAP based feature importance '''
    def cross_gen_shap_initial_setup(self, model_ = None, device_ = torch.device("cuda"), n_splits_ = 5, fold_index = 0):
        '''
        SHAP Feature Importance for cross generalization initial setup
        We will be using training and validation data from the first fold by default
        ''' 
        self.extract_train_test_common_PIDs()
        design()
        
        #Trial W for training 
        self.trial_train = self.labels[self.labels['scenario']==self.train_framework] #Full trial W with all 32 subjects 
        #Trial WT for testing 
        self.trial_test = self.labels[self.labels['scenario']==self.test_framework] #Full trial WT with all 26 subjects 
        
        #Training only data with strides from W
        train_only_trial_train = self.trial_train[self.trial_train.PID.isin(self.train_pids)] #subset of trial W with subjects only present in trial W but not in trial WT
        #Testing only data with strides from WT
        test_only_trial_test = self.trial_test[self.trial_test.PID.isin(self.test_pids)] #subset of trial WT with subjects only present in trial WT but not in trial W

        #Training data with strides from W for common PIDs in trials W and WT
        self.train_trial_train_commonPID = self.trial_train[self.trial_train.PID.isin(self.common_pids)] #subset of trial W with common subjects in trial W and WT
        #Testing data with strides from WT for common PIDs in trials W and WT
        self.test_trial_test_commonPID = self.trial_test[self.trial_test.PID.isin(self.common_pids)] #subset of trial W with common subjects in trial W and WT

        self.X_train_common = self.train_trial_train_commonPID.drop(['PID', 'label'], axis = 1)
        self.Y_train_common = self.train_trial_train_commonPID[['PID', 'label']]
        self.train_test_concatenated = self.labels[self.labels.scenario.isin([self.train_framework, self.test_framework])].reset_index().drop('index', axis = 1)    
        
        #Get dataloader 
        #Here the strides are normalized using within stride normalization, but frame counts are yet to normalized using training folds
        self.data = GaitDataset(self.data_path, self.labels_file, self.train_test_concatenated['PID'].unique(), framework = [self.train_framework, self.test_framework], datastream = "All")
        #Computing the X (91 features), Y (PID, label) for the models 
        self.X_sl = SliceDataset(self.data, idx = 0)
        self.Y_sl = SliceDataset(self.data, idx = 1)
        self.PID_sl = SliceDataset(self.data, idx = 2)
        
        #Shuffling the concatenated data
        self.train_test_concatenated, self.X_sl, self.Y_sl, self.PID_sl = shuffle(self.train_test_concatenated, self.X_sl, self.Y_sl, self.PID_sl, random_state = 0) 
        
        #Computing the training and test set indices for the CV folds         
        self.compute_train_test_indices_split(n_splits_)
        
        #Train and Test X and Y 
        train_indices = self.train_indices[fold_index]
        self.X_sl_train = self.X_sl[train_indices]
        self.Y_sl_train = self.Y_sl[train_indices]
        self.PID_sl_train = self.PID_sl[train_indices]
        
        test_indices = self.test_indices[fold_index]
        self.X_sl_test = self.X_sl[test_indices]
        self.Y_sl_test = self.Y_sl[test_indices]
        self.PID_sl_test = self.PID_sl[test_indices]
        
        #Model 
        self.create_folder_for_results()  
        self.torch_model = model_
        self.model = self.create_model(self.torch_model, device_)
        self.model.initialize() # This is important!
        self.model.load_params(f_params=self.save_results_path + self.parameter_dict['saved_model_path'], f_history = self.save_results_path + 'cnn1d_try_cross_gen_2021_05_25-23_57_22_107931\\train_end_history.json')
            
            
                 
class PermuteTransform():
    '''
    Class for permutation for features of interest for the testing folds with model trained on original features in the training folds 
    '''
    def __init__(self, feature_indices_):
        self.feature_indices_ = feature_indices_

    def permute_shuffle(self, x):
        '''
        Each element in the testing set has body coords for features of interest replaced with the shuffled version 
        '''
        for feat_index in self.feature_indices_:
            permuted_feature = self.X_shuffled[0]['body_coords'][:, feat_index]
            x['body_coords'][:, feat_index] = permuted_feature
        self.X_shuffled = self.X_shuffled[1:]
        return x

    def transform(self, X, y=None):
        '''
        Shuffle the desire features across the entire testing set 
        '''
        global runs
        np.random.seed(runs)
        #Create a shuffled copy for the testing dataset
        self.X_shuffled = shuffle(X, random_state = random.randint(0, 100))
        X.transform = self.permute_shuffle
        runs += 1
        return X
        
    def fit_transform(self, X, y=None):
        '''
        We do not need to shuffle the training set, so we return it as is
        '''
        return X

        
