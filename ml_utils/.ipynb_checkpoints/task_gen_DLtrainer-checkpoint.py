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
from ml_utils.DLutils import save_model, load_model

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
        self.save_results_path = self.parameter_dict['results_path'] + self.framework + '\\' + self.parameter_dict['model_path']
        self.save_results_prefix = self.parameter_dict['prefix_name'] + '_'
        self.save_results = self.parameter_dict['save_results']
        self.config_path = config_path
        self.re_train_epochs = self.parameter_dict['re_train_epochs']
        
                        
    
    def list_subjects_common_across_train_test(self):
        '''
        Since we need to implement pure task generalization framework, we must have same subjects across both training and testing trails 
        Hence, if there are some subjects that are present in the training set but not in the test set or vice versa, we eliminate those 
        subjects to have only common subjects across training and test sets. 
        Arguments: data subset for training and testing trial
        Returns: PIDs to retain in the training and test subsets with common subjects 
        '''

        print ('Original number of subjects in training and test sets:', len(self.trial_train['PID'].unique()), len(self.trial_test['PID'].unique()))

        #Try to use same subjects in trials W and WT for testing on same subjects we train on
        print ('Subjects in test set, which are not in training set')
        pids_missing_training = [] #PIDs missing in training set (trial W) but are present in the test set (trial WT)
        for x in self.trial_test['PID'].unique():
            if x not in self.trial_train['PID'].unique():
                pids_missing_training.append(x)
        print (pids_missing_training)
        #List of PIDs to retain in the training set 
        self.pids_retain_train = [i for i in self.trial_test['PID'].unique() if i not in pids_missing_training]

        print ('Subjects in training set, which are not in test set')
        pids_missing_test = [] #PIDs missing in test set (trial WT) but are present in the training set (trial W)
        for x in self.trial_train['PID'].unique():
            if x not in self.trial_test['PID'].unique():
                pids_missing_test.append(x)
        print (pids_missing_test)
        #List of PIDs to retain in the testing set 
        self.pids_retain_test = [i for i in self.trial_train['PID'].unique() if i not in pids_missing_test]

        print ('Number of subjects in training and test sets after reduction:', len(self.pids_retain_train), \
               len(self.pids_retain_test))

    
    
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
        #Task generalization W-> WT framework 
        #Loading the full training data in one go to compute the training data's mean and standard deviation for normalization 
        #We set the batch_size = len(training_data) for the same 
        training_data = GaitDataset(self.data_path, self.labels_file, self.pids_retain_train, framework = self.train_framework)   
        training_data_loader = DataLoader(training_data, batch_size = len(training_data), shuffle = self.parameter_dict['shuffle'], \
                                          num_workers = self.parameter_dict['num_workers'])
        #Since we loaded all the training data in a single batch, we can read all data and target in one go
        data, target, pid = next(iter(training_data_loader))
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
        self.X_sl_train, self.Y_sl_train, self.PID_sl_train = shuffle(self.X_sl_train, self.Y_sl_train, self.PID_sl_train, \
                                                                      random_state = 0) 
        
        self.X_sl_train_ = [x for x in self.X_sl_train]
#         print ('X format: ', self.X_sl_train_[0])
        self.Y_sl_train_ = [int(y) for y in self.Y_sl_train]
        self.PID_sl_train_ = [int(pid) for pid in self.PID_sl_train]
        
#         print ('Shuffled PID and Y:', self.PID_sl_train_, self.Y_sl_train_)
        
        gkf = StratifiedGroupKFold(n_splits=n_splits_)                 
        self.grid_search = GridSearchCV(self.model, param_grid = self.hyperparameter_grid, scoring = DLutils.accuracy_score_multi_class, \
                                        n_jobs = 1, cv = gkf.split(self.X_sl_train_, self.Y_sl_train_, groups=self.PID_sl_train_), \
                                        refit = True, return_train_score = True)
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
        self.grid_search.fit(self.X_sl_train, self.Y_sl_train, groups=self.PID_sl_train)
        end_time = time.time()
        self.training_time = end_time - start_time
        print("\nTraining/ Cross validation time: ", self.training_time)


    def resume_train(self):
        '''
        Resumes training of the skorch model for task generalization
        Arguments:
            model: Skorch model
            fullTrainLabelsList: List of training data labels and PIDs
            trainStridesList_norm: Normlized list of training sequences
            params: List of hyperparameters to optimize across
        Returns:
            Trained and tuned grid search object
        '''
        #shuffle data first
        self.X_sl_train, self.Y_sl_train, self.PID_sl_train = shuffle(self.X_sl_train, self.Y_sl_train, self.PID_sl_train, \
                                                                      random_state = 0) 

        print ('History:\n')
        display (pd.DataFrame(self.best_model.history))
        #Skorch callback history to get loss to plot
        start_time = time.time()
        self.best_model.fit(self.X_sl_train, self.Y_sl_train, epochs = self.re_train_epochs)
        end_time = time.time()
        self.training_time = end_time - start_time
        print("\nRe-Training time: ", self.training_time)

        
        
    def create_folder_for_results(self):
          #Create folder for saving results
        time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
        self.save_path = self.save_results_path + self.save_results_prefix + time_now+"\\"
        print("save path: ", self.save_path)
        os.mkdir(self.save_path)
        #Copy config file to results folder
        with open(self.config_path, 'rb') as src, open(self.save_path+"config.json", 'wb') as dst: dst.write(src.read())

            
    
    def learning_curves(self):
        '''
        To plot the training/validation loss and accuracy (stride-wise) curves over epochs 
        '''
        model_history = self.grid_search.best_estimator_.history
        epochs = [i for i in range(len(model_history))] #start from 1 instead of zero
#         print (epochs)
        train_loss = model_history[:,'train_loss']
#         print (train_loss)
        valid_loss = model_history[:,'valid_loss']
        train_acc = model_history[:,'train_acc']
        valid_acc = model_history[:,'valid_acc']
        #print("epochs", epochs, len(epochs))
        #print("train_acc", train_acc, len(train_acc))
        #print("train_loss", train_loss, len(train_loss))x
        #print("valid_loss", valid_loss, len(valid_loss))
        plt.plot(epochs,train_loss,'g*-'); #Dont print the last one for 3 built in
        plt.plot(epochs,valid_loss,'r*-');
        try:
            plt.plot(epochs,train_acc,'bo-');
        except:
            plt.plot(epochs,train_acc[:-1],'bo-');
        #plt.plot(np.arange(len(train_acc)),train_acc, 'b-'); #epochs and train_acc are off by one
        plt.plot(epochs,valid_acc, 'mo-');
        plt.title('Training/Validation loss and accuracy Curves');
        plt.xlabel('Epochs');
        plt.ylabel('Cross entropy loss/Accuracy');
        plt.legend(['Train loss','Validation loss', 'Train Accuracy', 'Validation Accuracy']); 
        if self.save_results:
            plt.savefig(self.save_path + 'learning_curve', dpi = 350)
        plt.show()
        plt.close()
    
    
    def evaluate(self):
        '''
        Function to evaluate ML models and plot it's confusion matrix
        Arguments: 
            model, test set, true test set labels, framework name, model name
            Computes the stride and subject wise test set evaluation metrics 
        Returns: 
            Prediction probabilities for HOA/MS/PD and stride and subject wise evaluation metrics 
            (Accuracy, Precision, Recall, F1 and AUC)
        '''
        true_labels = [int(y) for y in self.Y_sl_test] #Dropping the PID
    #     print ('Test labels', true_labels)
        
        eval_start_time = time.time()
        prediction_probs = self.best_model.predict(self.X_sl_test)
        eval_end_time = time.time()
        self.eval_time = eval_end_time - eval_start_time
        print("\nEvaluation time: ", self.eval_time)
        prediction_labels = prediction_probs.argmax(axis = 1)
    #     print ('Predictions', prediction_labels)

        #Stride wise metrics 
        acc = accuracy_score(true_labels, prediction_labels)
        #For multiclass predictions, we need to use marco/micro average
        p_macro = precision_score(true_labels, prediction_labels, average='macro')  
        r_macro = recall_score(true_labels, prediction_labels, average = 'macro')
        f1_macro = f1_score(true_labels, prediction_labels, average= 'macro')
        #Micro metrics 
        p_micro = precision_score(true_labels, prediction_labels, average='micro')  
        r_micro = recall_score(true_labels, prediction_labels, average = 'micro')
        f1_micro = f1_score(true_labels, prediction_labels, average= 'micro')   
        #Weighted metrics 
        p_weighted = precision_score(true_labels, prediction_labels, average='weighted')  
        r_weighted = recall_score(true_labels, prediction_labels, average = 'weighted')
        f1_weighted = f1_score(true_labels, prediction_labels, average= 'weighted')      
        #Metrics for each class
        p_class_wise = precision_score(true_labels, prediction_labels, average=None)
        r_class_wise = recall_score(true_labels, prediction_labels, average = None)
        f1_class_wise = f1_score(true_labels, prediction_labels, average= None)

        #For computing the AUC, we would need prediction probabilities for all the 3 classes 
        #Macro
        auc_macro = roc_auc_score(true_labels, prediction_probs, multi_class = 'ovo', average= 'macro')
        #Micro
        try:
            auc_micro = roc_auc_score(true_labels, prediction_probs, multi_class = 'ovo', average= 'micro')
        except:
            auc_micro = None
        #Weighted 
        auc_weighted = roc_auc_score(true_labels, prediction_probs, multi_class = 'ovo', average= 'weighted')
        #For each class
        try:
            auc_class_wise = roc_auc_score(true_labels, prediction_probs, multi_class = 'ovo', average= None)
        except:
            auc_class_wise = None

        print('Stride-based model performance (Macro): ', acc, p_macro, r_macro, f1_macro, auc_macro)
        print('Stride-based model performance (Micro): ', acc, p_micro, r_micro, f1_micro, auc_micro)
        print('Stride-based model performance (Weighted): ', acc, p_weighted, r_weighted, f1_weighted, auc_weighted)
        print ('Stride-based model performance (Class-wise): ', acc, p_class_wise, r_class_wise, f1_class_wise, auc_class_wise)

        #For computing person wise metrics 
        temp = pd.DataFrame(data = np.array((self.PID_sl_test, true_labels)).T, columns = ['PID', 'label']) #True label for the stride 
        self.trueY = copy.deepcopy(temp)
        temp['pred'] = prediction_labels #Predicted label for the stride 
        #Saving the stride wise true and predicted labels for calculating the stride wise confusion matrix for each model
        if self.save_results:
            temp.to_csv(self.save_path + 'stride_wise_predictions_' + self.framework + '.csv')

        x = temp.groupby('PID')['pred'].value_counts().unstack()
        #Input for subject wise AUC is probabilities at columns [0, 1, 2]
        proportion_strides_correct = pd.DataFrame(columns = [0, 1, 2])
        probs_stride_wise = x.divide(x.sum(axis = 1), axis = 0).fillna(0)
        proportion_strides_correct[probs_stride_wise.columns] = probs_stride_wise
        proportion_strides_correct.fillna(0, inplace=True)
        proportion_strides_correct['True Label'] = self.trueY.groupby('PID').first()
        #Input for precision, recall and F1 score
        proportion_strides_correct['Predicted Label'] = proportion_strides_correct[[0, 1, 2]].idxmax(axis = 1) 
        #Saving the person wise true and predicted labels for calculating the subject wise confusion matrix for each model
        if self.save_results:
            proportion_strides_correct.to_csv(self.save_path  + 'person_wise_predictions_' + self.framework + '.csv')
        try:
            print ('Best model: ', self.best_model)
        except:
            pass
        #Person wise metrics 
        person_acc = accuracy_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'])
        #Macro metrics 
        person_p_macro = precision_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   average = 'macro')
        person_r_macro = recall_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                average = 'macro')
        person_f1_macro = f1_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                             average = 'macro')
        person_auc_macro = roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct[[0, 1, 2]], \
                                   multi_class = 'ovo', average= 'macro')
        #Micro metrics 
        person_p_micro = precision_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   average = 'micro')
        person_r_micro = recall_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                average = 'micro')
        person_f1_micro = f1_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                             average = 'micro')
        try:
            person_auc_micro = roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct[[0, 1, 2]], \
                                   multi_class = 'ovo', average= 'micro')
        except:
            person_auc_micro = None
        #Weighted metrics 
        person_p_weighted = precision_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   average = 'weighted')
        person_r_weighted = recall_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                average = 'weighted')
        person_f1_weighted = f1_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                             average = 'weighted')
        person_auc_weighted = roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct[[0, 1, 2]], \
                                   multi_class = 'ovo', average= 'weighted')    
        #Class-wise metrics 
        person_p_class_wise = precision_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   average = None)
        person_r_class_wise = recall_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                average = None)
        person_f1_class_wise = f1_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                             average = None)
        try:
            person_auc_class_wise = roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct[[0, 1, 2]], \
                                   multi_class = 'ovo', average= None)   
        except:
            person_auc_class_wise = None

        print('Person-based model performance (Macro): ', person_acc, person_p_macro, person_r_macro, person_f1_macro, person_auc_macro)
        print('Person-based model performance (Micro): ', person_acc, person_p_micro, person_r_micro, person_f1_micro, person_auc_micro)
        print('Person-based model performance (Weighted): ', person_acc, person_p_weighted, person_r_weighted, person_f1_weighted, person_auc_weighted)
        print('Person-based model performance (Class-wise): ', person_acc, person_p_class_wise, person_r_class_wise, person_f1_class_wise, person_auc_class_wise)
        print ('********************************')
        
        #Plotting and saving the stride and subject wise confusion matrices 
        #Stride wise confusion matrix
        plt.figure()
        confusion_matrix = pd.crosstab(temp['label'], temp['pred'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
        if self.save_results:
            plt.savefig(self.save_path +'CFmatrix_task_generalize_' + self.framework + '_stride_wise.png', dpi = 350)
        plt.show()
        plt.close()


        #Plotting and saving the subject wise confusion matrix 
        plt.figure()
        confusion_matrix = pd.crosstab(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                       rownames=['Actual'], colnames=['Predicted'], margins = True)
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
        if self.save_results:
            plt.savefig(self.save_path +'CFmatrix_task_generalize_' + self.framework + '.png', dpi = 350)
        plt.show()
        plt.close()
        
        #For storing predicted probabilities for person (for all classes HOA/MS/PD) to show ROC curves 
        self.predicted_probs_person = pd.DataFrame(columns = [self.save_results_prefix  + cohort for cohort in ['HOA', 'MS', 'PD'] ]) 
        self.predicted_probs_person[self.save_results_prefix+'HOA'] = proportion_strides_correct[0]
        self.predicted_probs_person[self.save_results_prefix+'MS'] = proportion_strides_correct[1]
        self.predicted_probs_person[self.save_results_prefix+'PD'] = proportion_strides_correct[2]
        
        if self.save_results:
            self.predicted_probs_person.to_csv(self.save_path + 'task_generalize_'+self.framework+'_prediction_probs.csv')
        
        self.metrics = pd.DataFrame(columns = [self.save_results_prefix]) #Dataframe to store accuracies for each ML model for raw data
        try:
            best_parameters = self.grid_search.best_params_
        except:
            best_parameters = self.best_model.get_params()
            
        self.metrics[self.save_results_prefix] = [acc, p_macro, p_micro, p_weighted, p_class_wise, r_macro, r_micro, r_weighted, r_class_wise, f1_macro, f1_micro, f1_weighted, f1_class_wise, auc_macro, auc_micro, auc_weighted, auc_class_wise, person_acc, person_p_macro, person_p_micro, person_p_weighted, person_p_class_wise, person_r_macro, person_r_micro, person_r_weighted, person_r_class_wise, person_f1_macro, person_f1_micro, person_f1_weighted, person_f1_class_wise, person_auc_macro, person_auc_micro, person_auc_weighted, person_auc_class_wise, self.training_time, self.eval_time, self.total_parameters, self.trainable_params, self.nontrainable_params, best_parameters]                                  
                                    
        self.metrics.index = ['stride_accuracy', 'stride_precision_macro', 'stride_precision_micro', 'stride_precision_weighted', \
                 'stride_precision_class_wise', 'stride_recall_macro', 'stride_recall_micro', \
                 'stride_recall_weighted', 'stride_recall_class_wise', \
                 'stride_F1_macro', 'stride_F1_micro', 'stride_F1_weighted', 'stride_F1_class_wise', \
                 'stride_AUC_macro', 'stride_AUC_micro', 'stride_AUC_weighted',\
                 'stride_AUC_class_wise', 'person_accuracy', 'person_precision_macro', 'person_precision_micro', \
                 'person_precision_weighted', \
                 'person_precision_class_wise', 'person_recall_macro', 'person_recall_micro', \
                 'person_recall_weighted', 'person_recall_class_wise', \
                 'person_F1_macro', 'person_F1_micro', 'person_F1_weighted', 'person_F1_class_wise', \
                 'person_AUC_macro', 'person_AUC_micro', 'person_AUC_weighted', 'person_AUC_class_wise', 'cross validation time',\
                              'eval time', 'Model Parameters', 'Trainable Parameters', 'Nontrainable Parameters',\
                              'Best Parameters']  
        if self.save_results:
            self.metrics.to_csv(self.save_path + 'task_generalize_' + self.framework + '_result_metrics.csv')


    #Test set ROC curves for cohort prediction 
    def plot_ROC(self):
        '''
        Function to plot the ROC curve for models given in ml_models list 
        Arguments: 
            ml_models (name of models to plot the ROC for),  
            test_Y (true test set labels with PID), 
            predicted_probs_person (predicted test set probabilities for all 3 classes - HOA/MS/PD), 
            framework (WtoWT / VBWtoVBWT)
            save_results: Whether to save the results or not 
        Plots and saves the ROC curve with individual class-wise plots and micro/macro average plots 
        '''
        n_classes = 3 #HOA/MS/PD
        cohort = ['HOA', 'MS', 'PD']
        #PID-wise true labels 
        person_true_labels = self.trueY.groupby('PID').first()
        #Binarizing/getting dummies for the true labels i.e. class 1 is represented as 0, 1, 0
        person_true_labels_binarize = pd.get_dummies(person_true_labels.values.reshape(1, -1)[0])  

        sns.despine(offset=0)
        linestyles = ['-', '-', '-', '-.', '--', '-', '--', '-', '--']
        colors = ['b', 'magenta', 'cyan', 'g',  'red', 'violet', 'lime', 'grey', 'pink']
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(6, 4.5))
        axes.plot([0, 1], [0, 1], linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
        # person-based prediction probabilities for class 0: HOA, 1: MS, 2: PD
#         print ('self.predicted_probs_person', self.predicted_probs_person)
        model_probs = self.predicted_probs_person[[self.save_results_prefix+'HOA', self.save_results_prefix+'MS', \
                                                   self.save_results_prefix+'PD']]

        for i in range(n_classes): #For 3 classes 0, 1, 2
            fpr[i], tpr[i], _ = roc_curve(person_true_labels_binarize.iloc[:, i], model_probs.iloc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i]) #Computing the AUC score for each class
            #Plotting the ROCs for the three classes separately
            axes.plot(fpr[i], tpr[i], label = cohort[i] +' ROC (AUC = '+ str(round(roc_auc[i], 3))
                +')', linewidth = 3, alpha = 0.8, linestyle = linestyles[i], color = colors[i]) 

        # Compute micro-average ROC curve and ROC area (AUC)
        fpr["micro"], tpr["micro"], _ = roc_curve(person_true_labels_binarize.values.ravel(), model_probs.values.ravel())
        #Micro average AUC of ROC value
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
        axes.set_title('Task generalization '+ self.framework + ' '+ self.save_results_prefix)
        plt.legend()
        # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

        axes.set_xlabel('False Positive Rate')
        plt.tight_layout()
        if self.save_results:
            plt.savefig(self.save_path + 'ROC_task_generalize_' + self.framework + '.png', dpi = 350)
        plt.show()
        plt.close()

        
    
    def task_gen_setup(self, model_ = None, device_ = torch.device("cuda"), n_splits_ = 5):
        #Task generalization W-> WT framework 
        #Trial W for training 
        self.trial_train = self.labels[self.labels['scenario']==self.train_framework]
        #Trial WT for testing 
        self.trial_test = self.labels[self.labels['scenario']==self.test_framework]
        #Returning the PIDs of common subjects in training and testing set
        self.list_subjects_common_across_train_test()
        #Note that both pids_retain_trialW, pids_retain_trialWT will be the same since we are only retaining common subjects 
        #in training and testing trials for a "pure" task generalization framework

        #Showing the statistics and imbalance ratio of training and testing data 
        trial_train_reduced = self.trial_train[self.trial_train.PID.isin(self.pids_retain_train)]
        print ('Strides in training set: ', trial_train_reduced.shape[0])
        print ('HOA, MS and PD strides in training set:\n', trial_train_reduced['cohort'].value_counts())

        trial_test_reduced = self.trial_test[self.trial_test.PID.isin(self.pids_retain_test)]
        print ('Strides in testing set: ', trial_test_reduced.shape[0])
        print ('HOA, MS and PD strides in testing set:\n', trial_test_reduced['cohort'].value_counts())
        print ('Imbalance ratio (controls:MS:PD)= 1:X:Y\n', trial_test_reduced['cohort'].value_counts()/trial_test_reduced['cohort'].value_counts()['HOA'])
        
        self.get_data_loaders()
        self.create_folder_for_results()   
    
        if self.parameter_dict['behavior'] == 'train':
            self.model = self.create_model(model_, device_)
            self.X_sl_train = SliceDataset(self.training_data, idx = 0)
            self.Y_sl_train = SliceDataset(self.training_data, idx = 1)
            self.PID_sl_train = SliceDataset(self.training_data, idx = 2)
            self.train(n_splits_)
            cv_results = pd.DataFrame(self.grid_search.cv_results_)
            if self.save_results:
                cv_results.to_csv(self.save_path+"cv_results.csv")
            self.learning_curves()
            print("\nBest parameters: ", self.grid_search.best_params_)
            if self.save_results:
                best_params_file = open(self.save_path + "best_parameters.txt","w") 
                best_params_file.write(str(self.grid_search.best_params_))
                best_params_file.close() 
            self.best_model = self.grid_search.best_estimator_
            if self.save_results:
                save_model(self.best_model, self.save_path)
        
        if self.parameter_dict['behavior'] == 'evaluate':
            self.training_time = 0
            self.best_model = load_model(self.save_results_path + self.parameter_dict['saved_model_path'])
#             print (self.best_model.get_params())
#             display (pd.DataFrame(self.best_model.history))
        
        if self.parameter_dict['behavior'] == 'resume_training':
            self.best_model = load_model(self.save_results_path + self.parameter_dict['saved_model_path'])            
            self.X_sl_train = SliceDataset(self.training_data, idx = 0)
            self.Y_sl_train = SliceDataset(self.training_data, idx = 1)
            self.PID_sl_train = SliceDataset(self.training_data, idx = 2)
            self.resume_train()
            
        
        #Count of parameters in the selected model
        self.total_parameters = sum(p.numel() for p in self.best_model.module.parameters())        
        self.trainable_params =  sum(p.numel() for p in self.best_model.module.parameters() if p.requires_grad)
        self.nontrainable_params = self.total_parameters - self.trainable_params

        self.X_sl_test = SliceDataset(self.testing_data, idx = 0)
        self.Y_sl_test = SliceDataset(self.testing_data, idx = 1)
        self.PID_sl_test = SliceDataset(self.testing_data, idx = 2)
    
        self.evaluate() 
        self.plot_ROC()   
    