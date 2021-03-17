from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *
from ml_utils.split import StratifiedGroupKFold

def extract_train_test_common_PIDs(data, train_framework = 'W', test_framework = 'WT'):
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
    original_pids[train_framework] = data[data.scenario==train_framework].PID.unique()
    original_pids[test_framework] = data[data.scenario==test_framework].PID.unique()
    print ('Original number of subjects in training task', train_framework, 'are:', len(original_pids[train_framework]))
    print ('Original number of subjects in testing task', test_framework, 'are:', len(original_pids[test_framework]))
    
    #List of common PIDs across the train and test frameworks
    common_pids = list(set(original_pids[train_framework]) & set(original_pids[test_framework]))
    print ('Common number of subjects across train and test frameworks: ', len(common_pids))
    print ('Common subjects across train and test frameworks: ', common_pids)
    #List of PIDs only in the training set but not in the test set
    train_pids = list(set(original_pids[train_framework])^set(common_pids))
    print ('Number of subjects only in training framework: ', len(train_pids))
    print ('Subjects only in training framework: ', train_pids)
    #List of PIDs only in the testing set but not in the training set
    test_pids = list(set(original_pids[test_framework])^set(common_pids))
    print ('Number of subjects only in test framework: ', len(test_pids))
    print ('Subjects only in test framework: ', test_pids)    
    return train_pids, test_pids, common_pids



def compute_train_test_indices_split(train_test_concatenated, X_train_common, Y_train_common, train_pids, test_pids, \
                                     train_framework, test_framework):
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
    #List to append lists of training and test indices for each CV fold
    train_indices, test_indices = [], []
    #PIDs define the groups for stratified group 5-fold CV
    groups_ = Y_train_common['PID']

    #We use stratified group K-fold to sample our strides data
    gkf = StratifiedGroupKFold(n_splits=5) 

    #Indices for strides of subjects that exist only in training set 
    train_only_indices = train_test_concatenated[(train_test_concatenated.PID.isin(train_pids)) \
                                                 & (train_test_concatenated.scenario==train_framework)].index
    #Indices for strides of subjects that exist only in testing set 
    test_only_indices = train_test_concatenated[(train_test_concatenated.PID.isin(test_pids)) \
                                                 & (train_test_concatenated.scenario==test_framework)].index

    #Computing the CV fold indices for common subjects in training and testing 
    for train_idx, test_idx in gkf.split(X_train_common, Y_train_common['label'], groups=groups_):
        #PIDs for train indices using Stratified group 5-fold split
        train_split_pids = groups_.iloc[train_idx].unique()
    #     print ('train_pids', train_split_pids)
        #Indices for training using CV split in each fold 
        train_split_indices = train_test_concatenated[(train_test_concatenated.PID.isin(train_split_pids)) \
                                                     & (train_test_concatenated.scenario==train_framework)].index
        #Concatenating the indices of strides for PIDs in training only and CV split training PIDs 
        train_split_indices = train_split_indices.union(train_only_indices)
    #     print (train_split_indices, train_split_indices.shape)
        #Appending the training indices for the current fold 
        train_indices.append(train_split_indices)

        #PIDs for test indices using Stratified group 5-fold split
        test_split_pids = groups_.iloc[test_idx].unique()
    #     print ('test_pids', test_split_pids)
        #Indices for testing using CV split in each fold 
        test_split_indices = train_test_concatenated[(train_test_concatenated.PID.isin(test_split_pids)) \
                                                     & (train_test_concatenated.scenario==test_framework)].index
        #Concatenating the indices of strides for PIDs in testing only and CV split testing PIDs 
        test_split_indices = test_split_indices.union(test_only_indices)
    #     print (test_split_indices, test_split_indices.shape)
        #Appending the testing indices for the current fold 
        test_indices.append(test_split_indices)
    
    #Computing the .iloc indices from the .loc indices of the training and testing strides 
    train_indices = [train_test_concatenated.reset_index().index[train_test_concatenated.index.isin(train_indices[i])] for i in range(len(train_indices))]
    test_indices = [train_test_concatenated.reset_index().index[train_test_concatenated.index.isin(test_indices[i])] for i in range(len(test_indices))]
    return train_indices, test_indices



def evaluate(model, test_features, yoriginal_, ypredicted_, framework, model_name, results_path, save_results = True):
#     print ('yoriginal_', yoriginal_)
#     print ('ypredicted_', ypredicted_)
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
    test_subjects_true_predicted_labels = pd.DataFrame()
    
    best_index = model.cv_results_['mean_test_accuracy'].argmax()
    print('best_params: ', model.cv_results_['params'][best_index])

    n_folds = 5
    person_acc, person_p_macro, person_p_micro, person_p_weighted, person_p_class_wise, person_r_macro, person_r_micro, person_r_weighted, person_r_class_wise, person_f1_macro, person_f1_micro, person_f1_weighted, person_f1_class_wise, person_auc_macro, person_auc_weighted = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
    class_wise_scores = {'precision_class_wise': [], 'recall_class_wise': [], 'f1_class_wise': []}

    for i in range(n_folds):
        #For each fold, there are 2 splits: test and train (in order) and we need to retrieve the index 
        #of only test set for required 5 folds (best index)
        temp = test_features.loc[yoriginal_[(best_index*n_folds) + (i)].index] #True labels for the test strides in each fold
        temp['pred'] = ypredicted_[(best_index*n_folds) + (i)] #Predicted labels for the strides in the test set in each fold
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
        proportion_strides_correct['True Label'] = test_features.groupby('PID').first()
        #Input for precision, recall and F1 score
        proportion_strides_correct['Predicted Label'] = proportion_strides_correct[[0, 1, 2]].idxmax(axis = 1) 
        #Appending the test subjects' true and predicted label for each fold to compute subject-wise confusion matrix 
        test_subjects_true_predicted_labels = test_subjects_true_predicted_labels.append(proportion_strides_correct)          
            
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
    scores={'accuracy': make_scorer(acc), 'precision_macro':make_scorer(precision_score, average = 'macro'), \
            'precision_micro':make_scorer(precision_score, average = 'micro'), 'precision_weighted':make_scorer(precision_score, average = 'weighted'), 'precision_class_wise':[], 'recall_macro':make_scorer(recall_score, average = 'macro'), 'recall_micro':make_scorer(recall_score, average = 'micro'), 'recall_weighted':make_scorer(recall_score, average = 'weighted'), 'recall_class_wise': [], 'f1_macro': make_scorer(f1_score, average = 'macro'), 'f1_micro': make_scorer(f1_score, average = 'micro'), 'f1_weighted': make_scorer(f1_score, average = 'weighted'),
        'f1_class_wise': [], 'auc_macro': make_scorer(roc_auc_score, average = 'macro', multi_class = 'ovo', needs_proba= True), 'auc_weighted': make_scorer(roc_auc_score, average = 'weighted', multi_class = 'ovo', needs_proba= True)}
    
    for score in scores:
        try:
            stride_metrics_mean.append(model.cv_results_['mean_test_'+score][best_index])
            stride_metrics_std.append(model.cv_results_['std_test_'+score][best_index])
        except:
            stride_metrics_mean.append(list(map(mean, zip(*class_wise_scores[score]))))
            stride_metrics_std.append(list(map(stdev, zip(*class_wise_scores[score]))))
    print('\nStride-based model performance (mean): ', stride_metrics_mean)
    print('\nStride-based model performance (standard deviation): ', stride_metrics_std)
                                                                 
    print('\nPerson-based model performance (mean): ', person_means)
    print('\nPerson-based model performance (standard deviation): ', person_stds)
   
    #Saving the stride and person wise true and predicted labels for calculating the 
    #stride and subject wise confusion matrix for each model
    if save_results:
        test_strides_true_predicted_labels.to_csv(results_path+ framework + '\\stride_wise_predictions_' + \
                                      str(model_name) + '_' + framework + '.csv')
        test_subjects_true_predicted_labels.to_csv(results_path+ framework + '\\person_wise_predictions_' + \
                                      str(model_name) + '_' + framework + '.csv')
    
    
    #Plotting and saving the sequence and subject wise confusion matrices 
    #Sequence wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(test_strides_true_predicted_labels['label'], test_strides_true_predicted_labels['pred'], \
                               rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    if save_results:
        plt.savefig(results_path + framework + '\\CFmatrix_cross_generalize_' + framework + '_'+ str(model_name) + '_stride_wise.png', dpi = 350)
    plt.show()
    
    #Plotting and saving the subject wise confusion matrix 
    plt.figure()
    confusion_matrix = pd.crosstab(test_subjects_true_predicted_labels['True Label'], test_subjects_true_predicted_labels['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
    if save_results:
        plt.savefig(results_path + framework+'\\CFmatrix_cross_generalize_' + framework + '_'+ str(model_name) + '.png', dpi = 350)
    plt.show()
    
    
    return test_subjects_true_predicted_labels, [stride_metrics_mean, stride_metrics_std, person_means, person_stds]



def acc(y_true,y_pred):
    '''
    Returns the accuracy 
    Saves the true and predicted labels for training and test sets
    '''
    global yoriginal, ypredicted
    yoriginal.append(y_true)
    ypredicted.append(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy



#We do not use LDA/QDA since our features are not normally distributed 
def models(X, Y, train_indices, test_indices, model_name = 'random_forest', framework = 'W', results_path = '..\\MLresults\\', save_results = True):
    '''
    Arguments:
    X, Y, PID groups so that strides of each person are either in training or in testing set
    model: model_name, framework we wish to run the code for
    Returns: predicted probabilities and labels for each class, stride and subject based evaluation metrics 
    '''
    Y_ = Y['label'] #Dropping the PID
    groups_ = Y['PID']
    scores={'accuracy': make_scorer(acc), 'precision_macro':make_scorer(precision_score, average = 'macro'), \
            'precision_micro':make_scorer(precision_score, average = 'micro'), 'precision_weighted':make_scorer(precision_score, average = 'weighted'), 'recall_macro':make_scorer(recall_score, average = 'macro'), 'recall_micro':make_scorer(recall_score, average = 'micro'), 'recall_weighted':make_scorer(recall_score, average = 'weighted'), 'f1_macro': make_scorer(f1_score, average = 'macro'), 'f1_micro': make_scorer(f1_score, average = 'micro'), 'f1_weighted': make_scorer(f1_score, average = 'weighted'), 'auc_macro': make_scorer(roc_auc_score, average = 'macro', multi_class = 'ovo', needs_proba= True), 'auc_weighted': make_scorer(roc_auc_score, average = 'weighted', multi_class = 'ovo', needs_proba= True)}
    if(model_name == 'random_forest'): #Random Forest
        grid = {
       'randomforestclassifier__n_estimators': [40,45,50],\
       'randomforestclassifier__max_depth' : [15,20,25,None],\
       'randomforestclassifier__class_weight': [None, 'balanced'],\
       'randomforestclassifier__max_features': ['auto','sqrt','log2', None],\
       'randomforestclassifier__min_samples_leaf':[1,2,0.1,0.05]
        }
        #For z-score scaling on training and use calculated coefficients on test set
        rf_grid = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
        grid_search = GridSearchCV(rf_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
    
    if(model_name == 'adaboost'): #Adaboost
        ada_grid = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=0))
        grid = {
        'adaboostclassifier__n_estimators':[50, 75, 100, 125, 150],\
        'adaboostclassifier__learning_rate':[0.01,.1, 1, 1.5, 2]\
        }
        grid_search = GridSearchCV(ada_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
        
    if(model_name == 'kernel_svm'): #RBF SVM
        svc_grid = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', probability=True, random_state=0))
        grid = {
        'svc__gamma':[0.0001, 0.001, 0.1, 1, 10, ]\
        }
        grid_search = GridSearchCV(svc_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)

    if(model_name == 'gbm'): #GBM
        gbm_grid = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=0))
        grid = {
        'gradientboostingclassifier__learning_rate':[0.15,0.1,0.05], \
        'gradientboostingclassifier__n_estimators':[50, 100, 150],\
        'gradientboostingclassifier__max_depth':[2,4,7],\
        'gradientboostingclassifier__min_samples_split':[2,4], \
        'gradientboostingclassifier__min_samples_leaf':[1,3],\
        'gradientboostingclassifier__max_features':['auto','sqrt','log2', None],\
        }
        grid_search = GridSearchCV(gbm_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
    
    if(model_name=='xgboost'): #Xgboost
        xgb_grid = make_pipeline(StandardScaler(), xgboost.XGBClassifier(random_state=0))
        grid = {
            'xgbclassifier__min_child_weight': [1, 5],\
            'xgbclassifier__gamma': [0.1, 0.5, 1, 1.5, 2],\
            'xgbclassifier__subsample': [0.6, 0.8, 1.0],\
            'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],\
            'xgbclassifier__max_depth': [5, 7, 8]
        }
        grid_search = GridSearchCV(xgb_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
    
    if(model_name == 'knn'): #KNN
        knn_grid = make_pipeline(StandardScaler(), KNeighborsClassifier())
        grid = {
            'kneighborsclassifier__n_neighbors': [1, 3, 4, 5, 10],\
            'kneighborsclassifier__p': [1, 2, 3, 4, 5]\
        }
        grid_search = GridSearchCV(knn_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
        
    if(model_name == 'decision_tree'): #Decision Tree
        dec_grid = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
        #For z-score scaling on training and use calculated coefficients on test set
        grid = {'decisiontreeclassifier__min_samples_split': range(2, 50)}
        grid_search = GridSearchCV(dec_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)

    if(model_name == 'linear_svm'): #Linear SVM
        lsvm_grid = make_pipeline(StandardScaler(), SVC(kernel = 'linear', probability=True, random_state=0)) #LinearSVC(random_state=0, probability= True))
        grid = {
            'svc__gamma':[0.0001, 0.001, 0.1, 1, 10, ]\

        }
        grid_search = GridSearchCV(lsvm_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
    
    if(model_name == 'logistic_regression'): #Logistic regression
        lr_grid = make_pipeline(StandardScaler(), LogisticRegression())
        grid = {
            'logisticregression__random_state': [0]}
            
        grid_search = GridSearchCV(lr_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
    
    if(model_name == 'mlp'):
        mlp_grid = make_pipeline(StandardScaler(), MLPClassifier(random_state = 0, activation='relu', solver='adam',\
                                                       learning_rate = 'adaptive', learning_rate_init=0.001, 
                                                        shuffle=False, max_iter = 200))
        grid = {
            'mlpclassifier__hidden_layer_sizes': [(128, 8, 8, 128, 32), (50, 50, 50, 50, 50, 50, 150, 100, 10), 
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
        grid_search = GridSearchCV(mlp_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=zip(train_indices, test_indices), refit=False)
    grid_search.fit(X, Y_) #Fitting on the training set to find the optimal hyperparameters 
    test_subjects_true_predicted_labels, stride_person_metrics = evaluate(grid_search, Y, yoriginal, ypredicted, framework, model_name, results_path, save_results)
    return test_subjects_true_predicted_labels, stride_person_metrics



#ROC curves 
def plot_ROC(ml_model, test_set_true_predicted_labels, framework, results_path, save_results):
    '''
    Function to plot the ROC curve and confusion matrix for model given in ml_model name 
    Input: ml_models (name of models to plot the ROC for),  test_Y (true test set labels with PID), 
        predicted_probs_person (predicted test set probabilities for all 3 classes - HOA/MS/PD), framework (WtoWT / VBWtoVBWT)
    Plots and saves the ROC curve with individual class-wise plots and micro/macro average plots 
    '''
    n_classes = 3 #HOA/MS/PD
    cohort = ['HOA', 'MS', 'PD']
    ml_model_names = {'random_forest': 'RF', 'adaboost': 'AdaBoost', 'kernel_svm': 'RBF SVM', 'gbm': 'GBM', \
                  'xgboost': 'Xgboost', 'knn': 'KNN', 'decision_tree': 'DT',  'linear_svm': 'LSVM', 
             'logistic_regression': 'LR', 'mlp':'MLP'}

    #Binarizing/getting dummies for the true labels i.e. class 1 is represented as 0, 1, 0
    test_features_binarize = pd.get_dummies(test_set_true_predicted_labels['True Label'].values)     
    sns.despine(offset=0)
    linestyles = ['-', '-', '-', '-.', '--', '-', '--', '-', '--']
    colors = ['b', 'magenta', 'cyan', 'g',  'red', 'violet', 'lime', 'grey', 'pink']

    fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(6, 4.5))
    axes.plot([0, 1], [0, 1], linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
    # person-based prediction probabilities for class 0: HOA, 1: MS, 2: PD

    # Compute ROC curve and ROC area for each class
    tpr, fpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes): #n_classes = 3
        fpr[i], tpr[i], _ = roc_curve(test_features_binarize.iloc[:, i], test_set_true_predicted_labels.loc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #Plotting the ROCs for the three classes separately
        axes.plot(fpr[i], tpr[i], label = cohort[i] +' ROC (AUC = '+ str(round(roc_auc[i], 3))
            +')', linewidth = 3, alpha = 0.8, linestyle = linestyles[i], color = colors[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_features_binarize.values.ravel(),\
                                              test_set_true_predicted_labels[[0, 1, 2]].values.ravel())
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
    axes.set_title('Cross generalization '+framework + ' '+ ml_model_names[ml_model])
    plt.legend()
    # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

    axes.set_xlabel('False Positive Rate')
    plt.tight_layout()
    if save_results:
        plt.savefig(results_path + framework+'\\ROC_cross_generalize_' + framework + '_'+ ml_model+ '.png', dpi = 350)
    plt.show()
    
    
    
def run_ml_models(ml_models, X, Y, train_indices, test_indices, framework, results_path, save_results = True):
    '''
    Function to run the ML models for the required framework
    Arguments: 
        names of ml_models, X, Y, framework 
        save_results: Whether to save the csv files or not 
    Returns and saves .csv for evaluation metrics and tprs/fprs/rauc for the ROC curves 
    '''
    metrics = pd.DataFrame(columns = ml_models) #Dataframe to store accuracies for each ML model for raw data 
    for ml_model in ml_models:
        print (ml_model)
        global yoriginal, ypredicted
        yoriginal = []
        ypredicted = []
        test_subjects_true_predicted_labels, stride_person_metrics = models(X, Y, train_indices, test_indices, ml_model, framework, results_path, save_results)
        metrics[ml_model] = sum(stride_person_metrics, [])
        plot_ROC(ml_model, test_subjects_true_predicted_labels, framework, results_path, save_results)
        print ('********************************')
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
    
    
    metrics.index = [i + '_mean' for i in stride_scoring_metrics] + [i + '_std' for i in stride_scoring_metrics] + [i + '_mean' for i in person_scoring_metrics] + [i + '_std' for i in person_scoring_metrics]
    #Saving the evaluation metrics and tprs/fprs/rauc for the ROC curves 
    if save_results:
        metrics.to_csv(results_path+framework+'\\cross_generalize_'+framework+'_result_metrics.csv')
    return metrics



def design():
    print ('******************************************')