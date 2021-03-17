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
        self.framework = self.parameter_dict['framework'] #Defining the framework of interest  //"W" or "WT" or "reducedW_for_comparision_WandWTonly" or "reducedWT_for_comparision_WandWTonly" 
        #If "comparision" keyword exists, then we need to keep only common PIDs between the frameworks to compare 
        self.comparision_frameworks = self.parameter_dict['comparision_frameworks']
        self.scenario = self.parameter_dict['scenario']
        self.hyperparameter_grid = hyperparameter_grid
        self.save_results_path = self.parameter_dict['results_path']  + self.framework + '/'+ self.parameter_dict['model_path']
        self.save_results_prefix = self.parameter_dict['prefix_name'] + '_'
        self.save_results = self.parameter_dict['save_results']
        self.config_path = config_path
        
        
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
#         print ('In accuracy y true label: ', self.y_true_label)
        self.y_pred_label = self.y_pred.argmax(axis = 1)
#         print ('In accuracy y pred label: ', self.y_pred_label)
        self.yoriginal.append(self.y_true_label)
        self.ypredicted.append(self.y_pred_label)
#         print ('current self.yoriginal: ', self.yoriginal)
        accuracy = accuracy_score(self.y_true_label, self.y_pred_label)
#         print ('current accuracy: ', accuracy)
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
#         self.Y_sl = pd.DataFrame(self.Y_sl) #To attach indices for correspondance to relative PID
        
        self.X_sl_ = [x for x in self.X_sl]
#         print ('X format: ', self.X_sl_train_[0])
        self.Y_sl_ = [int(y) for y in self.Y_sl]
        self.PID_sl_ = [int(pid) for pid in self.PID_sl]
        
#         print ('Shuffled PID and Y:', self.PID_sl_, self.Y_sl_)        
        self.yoriginal, self.ypredicted = [], []
        self.pipe = Pipeline([('scale', custom_StandardScaler()), ('net', self.model)])
        
        self.gkf = StratifiedGroupKFold(n_splits=n_splits_)  
        self.scores = {'accuracy': self.accuracy_score_multi_class_cv, 'precision_macro': self.precision_macro_score_multi_class_cv, 'precision_micro': self.precision_micro_score_multi_class_cv, 'precision_weighted': self.precision_weighted_score_multi_class_cv, 'recall_macro': self.recall_macro_score_multi_class_cv, 'recall_micro': self.recall_micro_score_multi_class_cv, 'recall_weighted': self.recall_weighted_score_multi_class_cv, 'f1_macro': self.f1_macro_score_multi_class_cv, 'f1_micro': self.f1_micro_score_multi_class_cv, 'f1_weighted': self.f1_weighted_score_multi_class_cv, 'auc_macro': self.auc_macro_score_multi_class_cv, 'auc_weighted': self.auc_weighted_score_multi_class_cv}
        
        self.grid_search = GridSearchCV(self.pipe, param_grid = self.hyperparameter_grid, scoring = self.scores, \
                                        n_jobs = 1, cv = self.gkf.split(self.X_sl_, self.Y_sl_, groups=self.PID_sl_), \
                                        refit = False)
        print("grid search", self.grid_search)

#         print("Cross val split PIDs:\n")
#         for idx, (train, test) in enumerate(self.gkf.split(self.X_sl_, self.Y_sl_, groups=self.PID_sl_)):
#             print ('\nFold: ', idx+1)
#             print ('\nIndices in train: ', train, '\nIndices in test: ', test)
#             print('\nPIDs in TRAIN: ', np.unique(self.PID_sl[train], axis=0), '\nPIDs in TEST: ', \
#                   np.unique(self.PID_sl[test], axis=0))
#             print ('**************************************************************')

        #Skorch callback history to get loss to plot
        start_time = time.time()
        self.grid_search.fit(self.X_sl, pd.Series(self.Y_sl), groups = self.PID_sl)
        end_time = time.time()
        self.training_time = end_time - start_time
        print("\nTraining/ Cross validation time: ", self.training_time)
       
    
    def evaluate(self, n_splits_ = 5):
        '''
        Arguments: trained model, test set, true and predicted labels for test set, framework and model name 
        Returns: predicted probabilities and labels for each class, stride and subject based evaluation metrics 
        Saves the csv files for stride wise predictions and subject wise predictions for confusion matrix 
        '''
        #For creating the stride wise confusion matrix, we append the true and predicted labels for strides in each fold to this 
        #test_strides_true_predicted_labels dataframe 
        test_strides_true_predicted_labels = pd.DataFrame()
        #For creating the subject wise confusion matrix, we append the true and predicted labels for subjects in each fold to this
        #test_subjects_true_predicted_labels dataframe
        self.test_subjects_true_predicted_labels = pd.DataFrame()

        best_index = self.grid_search.cv_results_['mean_test_accuracy'].argmax()
    #     print (model.cv_results_)
        self.best_params = self.grid_search.cv_results_['params'][best_index]
        print('\nBest parameters: ', self.best_params)
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
    #     print (list(map(mean, zip(*person_p_class_wise))))
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
            plt.savefig(self.save_path + 'CFmatrix_subject_generalize_' + self.framework + '_stride_wise.png', dpi = 350)
        plt.show()

        #Plotting and saving the subject wise confusion matrix 
        plt.figure()
        confusion_matrix = pd.crosstab(self.test_subjects_true_predicted_labels['True Label'], self.test_subjects_true_predicted_labels['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
        if self.save_results:
            plt.savefig(self.save_path + 'CFmatrix_subject_generalize_' + self.framework + '.png', dpi = 350)
        plt.show()

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
        print ('metrics', self.metrics)
        print ('metrics index', [i + '_mean' for i in stride_scoring_metrics] + [i + '_std' for i in stride_scoring_metrics] + [i + '_mean' for i in person_scoring_metrics] + [i + '_std' for i in person_scoring_metrics] + ['training_time', 'best_parameters']  + ['total_parameters', 'trainable_params', 'nontrainable_params', 'Total Epochs'])
        self.metrics.index = [i + '_mean' for i in stride_scoring_metrics] + [i + '_std' for i in stride_scoring_metrics] + [i + '_mean' for i in person_scoring_metrics] + [i + '_std' for i in person_scoring_metrics] + ['training_time', 'best_parameters']  + ['total_parameters', 'trainable_params', 'nontrainable_params', 'Total Epochs']
    
        #Saving the evaluation metrics and tprs/fprs/rauc for the ROC curves 
        if self.save_results:
            self.metrics.to_csv(self.save_path + 'subject_generalize_'+self.framework+'_result_metrics.csv')

            

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
        axes.set_title('Subject generalization '+ self.framework)
        plt.legend()
        # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

        axes.set_xlabel('False Positive Rate')
        plt.tight_layout()
        if self.save_results:
            plt.savefig(self.save_path +'ROC_subject_generalize_' + self.framework + '_'+ self.save_results_prefix + '.png', dpi = 350)
        plt.show()


        
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

        for fold, (train_ix, val_ix) in enumerate(self.gkf.split(self.X_sl_, self.Y_sl_, groups=self.PID_sl_)):
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
        
            
            
    def subject_gen_setup(self, model_class = None, model = None, device_ = torch.device("cuda"), n_splits_ = 5, datastream = "All"):        
        self.device = device_
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
        self.data = GaitDataset(self.data_path, self.labels_file, self.trial['PID'].unique(), framework = self.scenario, datastream = datastream)      
        self.create_folder_for_results()   
        self.torch_model = model
        self.torch_model_class = model_class
        if self.parameter_dict['behavior'] == 'train':
            self.model = self.create_model(self.torch_model, device_)
            self.X_sl = SliceDataset(self.data, idx = 0)
            self.Y_sl = SliceDataset(self.data, idx = 1)
            self.PID_sl = SliceDataset(self.data, idx = 2)
            self.train(n_splits_)
            cv_results = pd.DataFrame(self.grid_search.cv_results_)
            if self.save_results:
                cv_results.to_csv(self.save_path+"cv_results.csv")
            self.learning_curves()    
        self.evaluate(n_splits_) 
        self.plot_ROC() 
