from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *
from ml_utils.split import StratifiedGroupKFold

def keep_subjects_common_across_train_test(trial_train, trial_test):
    '''
    Since we need to implement pure task generalization framework, we must have same subjects across both training and testing trails 
    Hence, if there are some subjects that are present in the training set but not in the test set or vice versa, we eliminate those 
    subjects to have only common subjects across training and test sets. 
    Arguments: data subset for training and testing trial
    Returns: training and testing subsets with common subjects 
    '''
    
    print ('Original number of subjects in training and test sets:', len(trial_train['PID'].unique()), len(trial_test['PID'].unique()))

    #Try to use same subjects in trials W and WT for testing on same subjects we train on
    print ('Subjects in test set, which are not in training set')
    pids_missing_training = [] #PIDs missing in training set (trial W) but are present in the test set (trial WT)
    for x in trial_test['PID'].unique():
        if x not in trial_train['PID'].unique():
            pids_missing_training.append(x)
    print (pids_missing_training)

    #Deleting the subjects from the test set that are missing in the training set 
    trial_test_reduced = trial_test.set_index('PID').drop(pids_missing_training).reset_index()

    print ('Subjects in training set, which are not in test set')
    pids_missing_test = [] #PIDs missing in test set (trial WT) but are present in the training set (trial W)
    for x in trial_train['PID'].unique():
        if x not in trial_test['PID'].unique():
            pids_missing_test.append(x)
    print (pids_missing_test)

    #Deleting the subjects from the training set that are missing in the test set 
    trial_train_reduced = trial_train.set_index('PID').drop(pids_missing_test).reset_index()

    print ('Number of subjects in training and test sets after reduction:', len(trial_train_reduced['PID'].unique()), \
           len(trial_test_reduced['PID'].unique()))
    #Returning the dataframes where the training and testing set have common subjects 
    return trial_train_reduced, trial_test_reduced 


#Standardize the data before ML methods 
#Take care that testing set is not used while normalizaing the training set, otherwise the train set indirectly contains 
#information about the test set
def normalize(dataframe, n_type):
    '''
    Function to compute the coefficients from the training data to normalize the training and test data sets 
    Arguments: 
        dataframe: dataframe to normalize
        n_type: type of normalization (z-score or min-max)
    Returns:
        Coefficients such that normalized_data = (data - mean)/sd
        mean: mean/min of the training data for z-score/min-max normalization respectively
        sd: standard deviation/max-min of the training data for z-score/min-max normalization respectively
    '''
    #col_names = list(dataframe.columns)
    if (n_type == 'z'): #z-score normalization
        mean = np.mean(dataframe)
        sd = np.std(dataframe)
    else: #min-max normalization
        mean = np.min(dataframe)
        sd = np.max(dataframe)-np.min(dataframe)
    return mean, sd



def models(trainX, trainY, testX, testY, model_name = 'random_forest', framework = 'WtoWT', results_path = '..\\MLresults\\', save_results = True, datastream_name = 'All'):
    '''
    Function to define and tune ML models 
    Arguments: 
        training set: trainX, testX, 
        testing set: testX, testY, 
        model: model_name, 
        framework
    Returns: 
        Prediction probabilities for HOA/MS/PD and stride and subject wise evaluation metrics 
        (Accuracy, Precision, Recall, F1 and AUC)
    Make sure the strides of same subject do not appear in both training and validation sets made out of trial W
    '''
    trainY1 = trainY['label'] #Dropping the PID
    #Make sure subjects are not mixed in training and validation sets, i.e. strides of same subject are either 
    #in training set or in validation set 
    groups_ = trainY['PID'] 
    #We use stratified group K-fold to sample our strides data
    gkf = StratifiedGroupKFold(n_splits=5) 
    
    if(model_name == 'random_forest'): #Random Forest
        grid = {
       'n_estimators': [40,45,50],\
       'max_depth' : [15,20,25,None],\
       'class_weight': [None, 'balanced'],\
       'max_features': ['auto','sqrt','log2', None],\
       'min_samples_leaf':[1,2,0.1,0.05]
        }
        rf_grid = RandomForestClassifier(random_state=0)
        #Make sure the strides of same subject do not appear in both training and validation sets made out of trial W
        grid_search = GridSearchCV(estimator = rf_grid, param_grid = grid, scoring='accuracy', n_jobs = 1, \
                                   cv=gkf.split(trainX, trainY1, groups=groups_))
    
    if(model_name == 'adaboost'): #Adaboost
        ada_grid = AdaBoostClassifier(random_state=0)
        grid = {
        'n_estimators':[50, 75, 100, 125, 150],\
        'learning_rate':[0.01,.1, 1, 1.5, 2]\
        }
        grid_search = GridSearchCV(ada_grid, param_grid = grid, scoring='accuracy', n_jobs = 1, \
                                   cv=gkf.split(trainX, trainY1, groups=groups_))
    
    if(model_name == 'kernel_svm'): #RBF SVM
        svc_grid = SVC(kernel = 'rbf', probability=True, random_state=0)
        grid = {
        'gamma':[0.0001, 0.001, 0.1, 1, 10, ]\
        }
        grid_search = GridSearchCV(svc_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))

    if(model_name == 'gbm'): #GBM
        gbm_grid = GradientBoostingClassifier(random_state=0)
        grid = {
        'learning_rate':[0.15,0.1,0.05], \
        'n_estimators':[50, 100, 150],\
        'max_depth':[2,4,7],\
        'min_samples_split':[2,4], \
        'min_samples_leaf':[1,3],\
        'max_features':[4, 5, 6]\
        }
        grid_search = GridSearchCV(gbm_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    
    if(model_name=='xgboost'): #Xgboost
        xgb_grid = xgboost.XGBClassifier(random_state=0)
        grid = {
            'min_child_weight': [1, 5],\
            'gamma': [0.1, 0.5, 1, 1.5, 2],\
            'subsample': [0.6, 0.8, 1.0],\
            'colsample_bytree': [0.6, 0.8, 1.0],\
            'max_depth': [5, 7, 8]
        }
        grid_search = GridSearchCV(xgb_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    
    if(model_name == 'knn'): #KNN
        knn_grid = KNeighborsClassifier()
        grid = {
            'n_neighbors': [1, 3, 4, 5, 10],\
            'p': [1, 2, 3, 4, 5]\
        }
        grid_search = GridSearchCV(knn_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
        
    if(model_name == 'decision_tree'): #Decision Tree
        dec_grid = DecisionTreeClassifier(random_state=0)
        grid = {
            'min_samples_split': range(2, 50),\
        }
        grid_search = GridSearchCV(dec_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    
    if(model_name == 'linear_svm'): #Linear SVM
        lsvm_grid = LinearSVC(random_state=0)
        grid = {
            'loss': ['hinge','squared_hinge']
        }
        grid_search = GridSearchCV(lsvm_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    
    if(model_name == 'logistic_regression'): #Logistic regression
        logistic_grid = LogisticRegression(random_state=0)
        grid = {
            'random_state': [0]
        }
        grid_search = GridSearchCV(logistic_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))  
    
    if(model_name == 'mlp'):
        mlp_grid = MLPClassifier(activation='relu', solver='adam', learning_rate = 'adaptive', learning_rate_init=0.001,\
                                                        shuffle=False, max_iter = 500, random_state = 0)
        grid = {
            'hidden_layer_sizes': [(128, 8, 8, 128, 32), (50, 50, 50, 50, 50, 50, 150, 100, 10), 
                                  (50, 50, 50, 50, 50, 60, 30, 20, 50), (50, 50, 50, 50, 50, 150, 10, 60, 150),
                                  (50, 50, 50, 50, 50, 5, 50, 10, 5), (50, 50, 50, 50, 50, 5, 50, 150, 150),
                                  (50, 50, 50, 50, 50, 5, 30, 50, 20), (50, 50, 50, 50, 10, 150, 20, 20, 30),
                                  (50, 50, 50, 50, 30, 150, 100, 20, 100), (50, 50, 50, 50, 30, 5, 100, 20, 100),
                                  (50, 50, 50, 50, 60, 50, 50, 60, 60), (50, 50, 50, 50, 20, 50, 60, 20, 20),
                                  (50, 50, 50, 10, 50, 10, 150, 60, 150), (50, 50, 50, 10, 50, 150, 30, 150, 5),
                                  (50, 50, 50, 10, 50, 20, 150, 5, 10), (50, 50, 50, 10, 150, 50, 20, 20, 100), 
                                  (50, 50, 50, 30, 100, 5, 30, 150, 30), (50, 50, 50, 50, 100, 150, 100, 200), 
                                  (50, 50, 50, 5, 5, 100, 100, 150), (50, 50, 5, 50, 200, 100, 150, 5), 
                                  (50, 50, 5, 5, 200, 100, 50, 30), (50, 50, 5, 10, 5, 200, 200, 10), 
                                  (50, 50, 5, 30, 5, 5, 50, 10), (50, 50, 5, 200, 50, 5, 5, 50), 
                                  (50, 50,50, 5, 5, 100, 100, 150), (5, 5, 5, 5, 5, 100, 50, 5, 50, 50), 
                                  (5, 5, 5, 5, 5, 100, 20, 100, 30, 30), (5, 5, 5, 5, 5, 20, 20, 5, 30, 100), 
                                  (5, 5, 5, 5, 5, 20, 20, 100, 10, 10), (5, 5, 5, 5, 10, 10, 30, 50, 10, 10), 
                                  (5, 5, 5, 5, 10, 100, 30, 30, 30, 10), (5, 5, 5, 5, 10, 100, 50, 10, 50, 10), 
                                  (5, 5, 5, 5, 10, 100, 20, 100, 30, 5), (5, 5, 5, 5, 30, 5, 20, 30, 100, 50), 
                                  (5, 5, 5, 5, 30, 100, 20, 50, 20, 30), (5, 5, 5, 5, 50, 30, 5, 50, 10, 100), 
                                  (21, 21, 7, 84, 21, 84, 84), (21, 21, 5, 42, 42, 7, 42), (21, 84, 7, 7, 7, 84, 5), 
                                  (21, 7, 84, 5, 5, 21, 120), (42, 5, 21, 21, 21, 5, 120), (42, 5, 42, 84, 7, 120, 84), 
                                  (50, 100, 10, 5, 100, 25), (10, 10, 25, 50, 25, 5), (50, 50, 50, 50, 50, 20, 30, 100, 60)]

        }
        grid_search = GridSearchCV(mlp_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
        
    #Making sure that strides of same subjects do not mix in training and validation sets 
    grid_search.fit(trainX, trainY1, groups=groups_) #Fitting on the training set to find the optimal hyperparameters 
#     print('best score: ', grid_search.best_score_)
#     print('best_params: ', grid_search.best_params_, grid_search.best_index_)
#     print('Mean cv accuracy on test set:', grid_search.cv_results_['mean_test_score'][grid_search.best_index_])
#     print('Standard deviation on test set:' , grid_search.cv_results_['std_test_score'][grid_search.best_index_])
#     print('Mean cv accuracy on train set:', grid_search.cv_results_['mean_train_score'][grid_search.best_index_])
#     print('Standard deviation on train set:', grid_search.cv_results_['std_train_score'][grid_search.best_index_])
#     print('Test set performance:\n')
    person_wise_prob_for_roc, stride_person_metrics = evaluate(grid_search, testX, testY, framework, model_name, results_path, save_results, datastream_name)
    return person_wise_prob_for_roc, stride_person_metrics



def evaluate(model, test_features, trueY, framework, model_name, results_path, save_results = True, datastream_name = 'All'):
    '''
    Function to evaluate ML models and plot it's confusion matrix
    Arguments: 
        model, test set, true test set labels, framework name, model name
        Computes the stride and subject wise test set evaluation metrics 
    Returns: 
        Prediction probabilities for HOA/MS/PD and stride and subject wise evaluation metrics 
        (Accuracy, Precision, Recall, F1 and AUC)
    '''
    test_labels = trueY['label'] #Dropping the PID
#     print ('Test labels', test_labels)
    predictions = model.predict(test_features)
#     print ('Predictions', predictions)
    
    #Stride wise metrics 
    acc = accuracy_score(test_labels, predictions)
    #For multiclass predictions, we need to use marco/micro average
    p_macro = precision_score(test_labels, predictions, average='macro')  
    r_macro = recall_score(test_labels, predictions, average = 'macro')
    f1_macro = f1_score(test_labels, predictions, average= 'macro')
    #Micro metrics 
    p_micro = precision_score(test_labels, predictions, average='micro')  
    r_micro = recall_score(test_labels, predictions, average = 'micro')
    f1_micro = f1_score(test_labels, predictions, average= 'micro')   
    #Weighted metrics 
    p_weighted = precision_score(test_labels, predictions, average='weighted')  
    r_weighted = recall_score(test_labels, predictions, average = 'weighted')
    f1_weighted = f1_score(test_labels, predictions, average= 'weighted')      
    #Metrics for each class
    p_class_wise = precision_score(test_labels, predictions, average=None)
    r_class_wise = recall_score(test_labels, predictions, average = None)
    f1_class_wise = f1_score(test_labels, predictions, average= None)
    
    try:
        prediction_prob = model.predict_proba(test_features) #Score of the class with greater label
#         print ('Prediction Probability', model.predict_proba(test_features))
        
    except:
        prediction_prob = model.best_estimator_._predict_proba_lr(test_features) #For linear SVM
#         print ('Prediction Probability', model.best_estimator_._predict_proba_lr(test_features))
    
    #For computing the AUC, we would need prediction probabilities for all the 3 classes 
    #Macro
    auc_macro = roc_auc_score(test_labels, prediction_prob, multi_class = 'ovo', average= 'macro')
    #Micro
    try:
        auc_micro = roc_auc_score(test_labels, prediction_prob, multi_class = 'ovo', average= 'micro')
    except:
        auc_micro = None
    #Weighted 
    auc_weighted = roc_auc_score(test_labels, prediction_prob, multi_class = 'ovo', average= 'weighted')
    #For each class
    try:
        auc_class_wise = roc_auc_score(test_labels, prediction_prob, multi_class = 'ovo', average= None)
    except:
        auc_class_wise = None
    
    print('Stride-based model performance (Macro): ', acc, p_macro, r_macro, f1_macro, auc_macro)
    print('Stride-based model performance (Micro): ', acc, p_micro, r_micro, f1_micro, auc_micro)
    print('Stride-based model performance (Weighted): ', acc, p_weighted, r_weighted, f1_weighted, auc_weighted)
    print ('Stride-based model performance (Class-wise): ', acc, p_class_wise, r_class_wise, f1_class_wise, auc_class_wise)
    
    #For computing person wise metrics 
    temp = copy.deepcopy(trueY) #True label for the stride 
    temp['pred'] = predictions #Predicted label for the stride 
    #Saving the stride wise true and predicted labels for calculating the stride wise confusion matrix for each model
    if save_results:
        temp.to_csv(results_path+ framework + '\\stride_wise_predictions_' + str(model_name) + '_'+ str(datastream_name) + '_' + framework + '.csv')
    
    x = temp.groupby('PID')['pred'].value_counts().unstack()
    #Input for subject wise AUC is probabilities at columns [0, 1, 2]
    proportion_strides_correct = pd.DataFrame(columns = [0, 1, 2])
    probs_stride_wise = x.divide(x.sum(axis = 1), axis = 0).fillna(0)
    proportion_strides_correct[probs_stride_wise.columns] = probs_stride_wise
    proportion_strides_correct.fillna(0, inplace=True)
    proportion_strides_correct['True Label'] = trueY.groupby('PID').first()
    #Input for precision, recall and F1 score
    proportion_strides_correct['Predicted Label'] = proportion_strides_correct[[0, 1, 2]].idxmax(axis = 1) 
    #Saving the person wise true and predicted labels for calculating the subject wise confusion matrix for each model
    if save_results:
        proportion_strides_correct.to_csv(results_path+ framework + '\\person_wise_predictions_' + \
                                      str(model_name) + '_' + str(datastream_name) + '_' + framework + '.csv')
    try:
        print (model.best_estimator_)
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
    
    #Plotting and saving the stride and subject wise confusion matrices 
    #Stride wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(temp['label'], temp['pred'], \
                               rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    if save_results:
        plt.savefig(results_path + framework +'\\CFmatrix_task_generalize_' + framework + '_'+ str(model_name) + '_'+ str(datastream_name) + '_stride_wise.png', dpi = 350)
    plt.show()

    
    #Plotting and saving the subject wise confusion matrix 
    plt.figure()
    confusion_matrix = pd.crosstab(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
    if save_results:
        plt.savefig(results_path + framework+'\\CFmatrix_task_generalize_' + framework + '_'+ str(model_name) + '_'+ str(datastream_name) + '.png', dpi = 350)
    plt.show()
    return proportion_strides_correct[[0, 1, 2]], [acc, p_macro, p_micro, p_weighted, p_class_wise, r_macro, r_micro, r_weighted, r_class_wise, f1_macro, f1_micro, f1_weighted, f1_class_wise, auc_macro, auc_micro, auc_weighted, auc_class_wise, person_acc, person_p_macro, person_p_micro, person_p_weighted, person_p_class_wise, person_r_macro, person_r_micro, person_r_weighted, person_r_class_wise, person_f1_macro, person_f1_micro, person_f1_weighted, person_f1_class_wise, person_auc_macro, person_auc_micro, person_auc_weighted, person_auc_class_wise] 



#Test set ROC curves for cohort prediction 
def plot_ROC(ml_models, testY, predicted_probs_person, framework, results_path, save_results = True, datastream_name = 'All'):
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
    ml_model_names = {'random_forest': 'RF', 'adaboost': 'Adaboost', 'kernel_svm': 'RBF SVM', 'gbm': 'GBM', \
                      'xgboost': 'Xgboost', 'knn': 'KNN', 'decision_tree': 'DT',  'linear_svm': 'LSVM', 
                 'logistic_regression': 'LR', 'mlp': 'MLP'}
    #PID-wise true labels 
    person_true_labels = testY.groupby('PID').first()
    #Binarizing/getting dummies for the true labels i.e. class 1 is represented as 0, 1, 0
    person_true_labels_binarize = pd.get_dummies(person_true_labels.values.reshape(1, -1)[0])  

    sns.despine(offset=0)
    linestyles = ['-', '-', '-', '-.', '--', '-', '--', '-', '--']
    colors = ['b', 'magenta', 'cyan', 'g',  'red', 'violet', 'lime', 'grey', 'pink']
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for idx, ml_model in enumerate(ml_models): #Plotting the ROCs for all models in ml_models list
        fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(6, 4.5))
        axes.plot([0, 1], [0, 1], linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
        # person-based prediction probabilities for class 0: HOA, 1: MS, 2: PD
        model_probs = predicted_probs_person[[ml_model+'_HOA', ml_model+'_MS', ml_model+'_PD']]

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
        axes.set_title('Task generalization '+framework + ' '+ ml_model_names[ml_model])
        plt.legend()
        # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

        axes.set_xlabel('False Positive Rate')
        plt.tight_layout()
        if save_results:
            plt.savefig(results_path + framework+'\\ROC_task_generalize_' + framework + '_'+ ml_model + '_'+ str(datastream_name) + '.png', dpi = 350)
        plt.show()