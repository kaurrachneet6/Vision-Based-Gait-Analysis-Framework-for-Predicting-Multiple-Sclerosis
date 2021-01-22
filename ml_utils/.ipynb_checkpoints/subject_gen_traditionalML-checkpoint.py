from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *
from ml_utils.split import StratifiedGroupKFold

def evaluate(model, test_features, yoriginal_, ypredicted_, framework, model_name, results_path, save_results = True):
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
    test_subjects_true_predicted_labels = pd.DataFrame()
    
    best_index = model.cv_results_['mean_test_accuracy'].argmax()
#     print (model.cv_results_)
    print('best_params: ', model.cv_results_['params'][best_index])
    
    n_folds = 5
    person_acc, person_p_macro, person_p_micro, person_p_weighted, person_p_class_wise, person_r_macro, person_r_micro, person_r_weighted, person_r_class_wise, person_f1_macro, person_f1_micro, person_f1_weighted, person_f1_class_wise, person_auc_macro, person_auc_weighted = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
    class_wise_scores = {'precision_class_wise': [], 'recall_class_wise': [], 'f1_class_wise': []}

    for i in range(n_folds):
        #For each fold, there are 2 splits: test and train (in order) and we need to retrieve the index 
        #of only test set for required 5 folds (best index)
        temp = test_features.loc[yoriginal_[(best_index*n_folds) + (i)].index] #True labels for the test strides in each fold
        temp['pred'] = ypredicted_[(best_index*n_folds) + (i)] #Predicted labels for the strides in the test set in each fold
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
#     print (list(map(mean, zip(*person_p_class_wise))))
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
        plt.savefig(results_path + framework + '\\CFmatrix_subject_generalize_' + framework + '_'+ str(model_name) + '_stride_wise.png', dpi = 350)
    plt.show()

    #Plotting and saving the subject wise confusion matrix 
    plt.figure()
    confusion_matrix = pd.crosstab(test_subjects_true_predicted_labels['True Label'], test_subjects_true_predicted_labels['Predicted Label'], \
                               rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
    if save_results:
        plt.savefig(results_path + framework + '\\CFmatrix_subject_generalize_' + framework + '_'+ str(model_name)+ '.png', dpi = 350)
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
def models(X, Y, model_name = 'random_forest', framework = 'W', results_path = '..\\MLresults\\', save_results = True):
    '''
    Arguments:
    X, Y, PID groups so that strides of each person are either in training or in testing set
    model: model_name, framework we wish to run the code for
    Returns: predicted probabilities and labels for each class, stride and subject based evaluation metrics 
    '''
    Y_ = Y['label'] #Dropping the PID
    groups_ = Y['PID']
    #We use stratified group K-fold to sample our strides data
    gkf = StratifiedGroupKFold(n_splits=5) 
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
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    if(model_name == 'adaboost'): #Adaboost
        ada_grid = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=0))
        grid = {
        'adaboostclassifier__n_estimators':[50, 75, 100, 125, 150],\
        'adaboostclassifier__learning_rate':[0.01,.1, 1, 1.5, 2]\
        }
        grid_search = GridSearchCV(ada_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
        
    if(model_name == 'kernel_svm'): #RBF SVM
        svc_grid = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', probability=True, random_state=0))
        grid = {
        'svc__gamma':[0.0001, 0.001, 0.1, 1, 10, ]\
        }
        grid_search = GridSearchCV(svc_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)

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
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
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
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    if(model_name == 'knn'): #KNN
        knn_grid = make_pipeline(StandardScaler(), KNeighborsClassifier())
        grid = {
            'kneighborsclassifier__n_neighbors': [1, 3, 4, 5, 10],\
            'kneighborsclassifier__p': [1, 2, 3, 4, 5]\
        }
        grid_search = GridSearchCV(knn_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
        
    if(model_name == 'decision_tree'): #Decision Tree
        dec_grid = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
        #For z-score scaling on training and use calculated coefficients on test set
        grid = {'decisiontreeclassifier__min_samples_split': range(2, 50)}
        grid_search = GridSearchCV(dec_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)

    if(model_name == 'linear_svm'): #Linear SVM
        lsvm_grid = make_pipeline(StandardScaler(), SVC(kernel = 'linear', probability=True, random_state=0)) #LinearSVC(random_state=0, probability= True))
        grid = {
            'svc__gamma':[0.0001, 0.001, 0.1, 1, 10, ]\

        }
        grid_search = GridSearchCV(lsvm_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    if(model_name == 'logistic_regression'): #Logistic regression
        lr_grid = make_pipeline(StandardScaler(), LogisticRegression())
        grid = {
            'logisticregression__random_state': [0]}
            
        grid_search = GridSearchCV(lr_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
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
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    grid_search.fit(X, Y_, groups=groups_) #Fitting on the training set to find the optimal hyperparameters 
    test_subjects_true_predicted_labels, stride_person_metrics = evaluate(grid_search, Y, yoriginal, ypredicted, framework, model_name, results_path, save_results)
    return test_subjects_true_predicted_labels, stride_person_metrics



#ROC curves 
def plot_ROC(ml_model, test_set_true_predicted_labels, framework, results_path, save_results = True):
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
    axes.set_title('Subject generalization '+framework + ' '+ ml_model_names[ml_model])
    plt.legend()
    # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

    axes.set_xlabel('False Positive Rate')
    plt.tight_layout()
    if save_results:
        plt.savefig(results_path + framework+'\\ROC_subject_generalize_' + framework + '_'+ ml_model+ '.png', dpi = 350)
    plt.show()
    
    
    
def run_ml_models(ml_models, X, Y, framework, results_path, save_results = True):
    '''
    Function to run the ML models for the required framework
    Arguments: names of ml_models, X, Y, framework 
    Returns and saves .csv for evaluation metrics and tprs/fprs/rauc for the ROC curves 
    save_results: Whether to save the csv files or not 
    '''
    metrics = pd.DataFrame(columns = ml_models) #Dataframe to store accuracies for each ML model for raw data 
    for ml_model in ml_models:
        print (ml_model)
        global yoriginal, ypredicted
        yoriginal = []
        ypredicted = []
        test_subjects_true_predicted_labels, stride_person_metrics = models(X, Y, ml_model, framework, results_path, save_results)
        metrics[ml_model] = sum(stride_person_metrics, [])
        plot_ROC(ml_model, test_subjects_true_predicted_labels, framework, results_path, save_results)
        print ('********************************')
    scoring_metrics = ['stride_mean_accuracy', 'stride_precision_macro', 'stride_precision_micro', 'stride_precision_weighted', \
                 'stride_precision_class_wise', 'stride_recall_macro', 'stride_recall_micro', \
                 'stride_recall_weighted', 'stride_recall_class_wise', \
                 'stride_F1_macro', 'stride_F1_micro', 'stride_F1_weighted', 'stride_F1_class_wise', \
                 'stride_AUC_macro', 'stride_AUC_weighted', 'person_accuracy', 'person_precision_macro', 'person_precision_micro', \
                 'person_precision_weighted', 'person_precision_class_wise', 'person_recall_macro', 'person_recall_micro', \
                 'person_recall_weighted', 'person_recall_class_wise', \
                 'person_F1_macro', 'person_F1_micro', 'person_F1_weighted', 'person_F1_class_wise', \
                 'person_AUC_macro', 'person_AUC_weighted']
    
    
    metrics.index = [i + '_mean' for i in scoring_metrics] + [i + '_std' for i in scoring_metrics]
    
    
    #Saving the evaluation metrics and tprs/fprs/rauc for the ROC curves 
    if save_results:
        metrics.to_csv(results_path+framework+'\\subject_generalize_'+framework+'_result_metrics.csv')
    return metrics



def design():
    print ('******************************************')
    
    

def keep_common_PIDs(data, frameworks = ['W', 'WT', 'SLW', 'SLWT']):
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
    for framework in frameworks:
        #Appending the original PIDs for each task
        original_pids[framework] = data[data.scenario==framework].PID.unique()
        print ('Original number of subjects in task', framework, 'are:', len(original_pids[framework]))

    #List of common PIDs across all frameworks
    common_pids = set(original_pids[frameworks[0]])
    for framework in frameworks[1:]:
        common_pids.intersection_update(original_pids[framework])
    common_pids = list(common_pids)
    print ('Common number of subjects across all frameworks: ', len(common_pids))
    print ('Common subjects across all frameworks: ', common_pids)
    return common_pids

