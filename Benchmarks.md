# Gait Video Study

### Single stride traditional ML-based benchmark results
* **Task generalization (W -> WT)**
    * Stride-wise metrics: 
        * Accuracy = 0.8307 (XGBoost)
        * Precision Macro = 0.835 (XGBoost), Precision Micro = 0.8307 (XGBoost), Precision Weighted = 0.834 (XGBoost), Precision class wise = [0.774, 0.832, 0.899] (XGBoost)
        * Recall Macro = 0.829 (XGBoost), Recall Micro = 0.8307 (XGBoost), Recall Weighted = 0.8307 (XGBoost), Recall class wise = [0.888, 0.716, 0.882] (XGBoost)
        * F1 Macro = 0.829 (XGBoost), F1 Micro = 0.8307 (XGBoost), F1 Weighted = 0.829 (XGBoost), F1 class wise = [0.827, 0.770, 0.890] (XGBoost)
        * AUC Macro = 0.945 (GBM), AUC Weighted = 0.945 (GBM)
    * Subject-wise metrics: 
        * Accuracy = 0.96 (GBM/Linear SVM)
        * Precision Macro = 0.962 (GBM/Linear SVM), Precision Micro = 0.96 (GBM/Linear SVM), Precision Weighted = 0.964 (GBM/Linear SVM), Precision class wise = [0.88, 1, 1] (GBM/Linear SVM)
        * Recall Macro = 0.962 (GBM/Linear SVM), Recall Micro = 0.96 (GBM/Linear SVM), Recall Weighted = 0.96 (GBM/Linear SVM), Recall class wise = [1, 0.88, 1] (GBM/Linear SVM)
        * F1 Macro = 0.960 (GBM/Linear SVM), F1 Micro = 0.96 (GBM/Linear SVM), F1 Weighted = 0.96 (GBM/Linear SVM), F1 class wise = [0.941, 0.941, 1] (GBM/Linear SVM)
        * AUC Macro = 1 (Linear SVM/GBM/XGBoost/Logistic Regression), AUC Weighted = 1 (Linear SVM/GBM/XGBoost/Logistic Regression)


* **Subject generalization (W)**
    * Stride-wise metrics: 
        * Accuracy = 0.576$`\pm`$0.06 (Logistic Regression)
        * Precision Macro = 0.565$`\pm`$0.06 (Logistic Regression), Precision Micro = 0.576 $`\pm`$ 0.06 (Logistic Regression), Precision Weighted = 0.60 $`\pm`$ 0.07, Precision class wise = [0.68$`\pm`$0.10, 0.517$`\pm`$0.12, 0.497$`\pm`$0.13] (Logistic Regression)
        * Recall Macro = 0.557$`\pm`$0.06 (Logistic Regression), Recall Micro = 0.576 $`\pm`$ 0.06 (Logistic Regression), Recall Weighted = 0.576 $`\pm`$ 0.06 (Logistic Regression), Recall class wise = [0.67 $`\pm`$ 0.16, 0.56 $`\pm`$ 0.12, 0.43 $`\pm`$ 0.17] (Logistic Regression)
        * F1 Macro = 0.542 $`\pm`$ 0.05 (Logistic Regression), F1 Micro = 0.576 $`\pm`$ 0.06 (Logistic Regression), F1 Weighted = 0.5709 $`\pm`$ 0.06 (Logistic Regression), F1 class wise = [0.66 $`\pm`$ 0.103, 0.534 $`\pm`$ 0.117, 0.424 $`\pm`$ 0.052] (Logistic Regression)
        * AUC Macro = 0.731 $`\pm`$ 0.046 (Logistic Regression), AUC Weighted = 0.7309 $`\pm`$ 0.047 (Logistic Regression)
    * Subject-wise metrics: 
        * Accuracy = 0.6904 $`\pm`$ 0.17 (Logistic Regression)
        * Precision Macro = 0.711 ± 0.177 (logistic_regression), Precision Micro = 0.69 ± 0.172 (logistic_regression), Precision Weighted = 0.832 ± 0.084 (linear_svm), Precision class wise = [0.833 ± 0.235, 0.2 ± 0.27, 0.2 ± 0.447] (kernel_svm)
        * Recall Macro = 0.7 ± 0.213 (logistic_regression), Recall Micro = 0.69 ± 0.172 (logistic_regression), Recall Weighted = 0.69 ± 0.172 (logistic_regression), Recall class wise = [0.866 ± 0.182, 0.633 ± 0.217, 0.6 ± 0.418] (logistic_regression)
        * F1 Macro = 0.671 ± 0.19 (logistic_regression), F1 Micro = 0.69 ± 0.172 (logistic_regression), F1 Weighted = 0.71 ± 0.157 (logistic_regression) , F1 class wise = [0.833 ± 0.235, 0.2 ± 0.273, 0.2 ± 0.447] (kernel_svm)
        * AUC Macro = 0.844 ± 0.104 (decision_tree), AUC Weighted = 0.843 ± 0.103 (decision_tree) 


* **Subject generalization (WT)**
    * Stride-wise metrics: 
        * Accuracy = 0.546±0.17 (mlp)
        * Precision Macro = 0.557±0.146 (mlp) , Precision Micro = 0.546±0.17 (mlp), Precision Weighted = 0.581±0.121 (mlp), Precision class wise = [0.469 ± 0.36, 0.243 ± 0.21, 0.718 ± 0.23] (kernel_svm)
        * Recall Macro = 0.547±0.16 (mlp), Recall Micro = 0.546±0.17 (mlp), Recall Weighted = 0.546±0.17 (mlp), Recall class wise = [0.515 ± 0.30, 0.399 ± 0.21, 0.613 ± 0.34] (adaboost)
        * F1 Macro = 0.523±0.175 (mlp), F1 Micro = 0.546±0.17 (mlp), F1 Weighted = 0.532±0.174 (mlp), F1 class wise = [0.445 ± 0.276, 0.429 ± 0.163, 0.552 ± 0.202] (adaboost)
        * AUC Macro = 0.734±0.162 (mlp), AUC Weighted = 0.738±0.156 (mlp)
    * Subject-wise metrics: 
        * Accuracy = 0.683±0.271 (mlp)
        * Precision Macro = 0.7±0.245 (mlp), Precision Micro = 0.683±0.271 (mlp), Precision Weighted = 0.892±0.082 (mlp), Precision class wise = [0.7 ± 0.44, 0.4 ± 0.547, 0.7 ± 0.447](adaboost)
        * Recall Macro = 0.667±0.337 (mlp), Recall Micro = 0.683±0.271 (mlp), Recall Weighted = 0.683±0.271 (mlp), Recall class wise = [0.533 ± 0.50, 0.8 ± 0.298, 0.666 ± 0.471](mlp)
        * F1 Macro = 0.64±0.311 (mlp), F1 Micro = 0.683±0.271 (mlp), F1 Weighted = 0.721±0.228 (mlp), F1 class wise = [0.7 ± 0.447, 0.4 ± 0.547, 0.7 ± 0.447] (adaboost)
        * AUC Macro = 0.917±0.059 (decision_tree), AUC Weighted = 0.91±0.065 (decision_tree)


* **Subject generalization (comparing W and WT: W)**
    * Stride-wise metrics: 
        * Accuracy = 0.541±0.117 (mlp)
        * Precision Macro = 0.561±0.202 (kernel_svm), Precision Micro = 0.541±0.117 (mlp), Precision Weighted = 0.574±0.216 (kernel_svm), Precision class wise = [0.60 ± 0.201, 0.474 ± 0.091, 0.527 ± 0.213] (logistic_regression)
        * Recall Macro = 0.556±0.136 (mlp), Recall Micro = 0.541±0.117 (mlp), Recall Weighted = 0.541±0.117 (mlp), Recall class wise = [0.675 ± 0.252, 0.518 ± 0.114, 0.474 ± 0.301] (mlp)
        * F1 Macro = 0.54±0.129 (mlp), F1 Micro = 0.541±0.117 (mlp), F1 Weighted = 0.534±0.117 (mlp), F1 class wise = [0.588 ± 0.11, 0.436 ± 0.115, 0.529 ± 0.269] (xgboost)
        * AUC Macro = 0.721±0.204 (kernel_svm), AUC Weighted = 0.718±0.203 (kernel_svm)
    * Subject-wise metrics: 
        * Accuracy = 0.643±0.125 (mlp)
        * Precision Macro = 0.667±0.149 (mlp), Precision Micro = 0.643±0.125 (mlp), Precision Weighted = 0.795±0.135 (decision_tree), Precision class wise = [0.8 ± 0.273, 0.7 ± 0.273, 0.5 ± 0.353] (mlp)
        * Recall Macro = 0.689±0.163 (mlp), Recall Micro = 0.643±0.125 (mlp), Recall Weighted = 0.643±0.125 (mlp), Recall class wise = [0.633 ± 0.217, 0.733 ± 0.252, 0.7 ± 0.447] (mlp)
        * F1 Macro = 0.64±0.149 (mlp), F1 Micro = 0.643±0.125 (mlp), F1 Weighted = 0.648±0.128 (mlp), F1 class wise = [0.8 ± 0.273, 0.7 ± 0.273, 0.5 ± 0.353] (mlp)
        * AUC Macro = 0.796±0.097 (linear_svm), AUC Weighted = 0.794±0.111 (linear_svm)

* **Subject generalization (comparing W and WT: WT)**
    * Stride-wise metrics: 
        * Accuracy = 0.534±0.137 (xgboost)
        * Precision Macro = 0.528±0.131 (xgboost), Precision Micro = 0.534±0.137 (xgboost), Precision Weighted = 0.568±0.111 (xgboost), Precision class wise = [0.411 ± 0.203, 0.532 ± 0.144, 0.635 ± 0.250] (decision_tree)
        * Recall Macro = 0.516±0.106 (decision_tree), Recall Micro = 0.534±0.137 (xgboost), Recall Weighted = 0.534±0.137 (xgboost), Recall class wise = [0.484 ± 0.33, 0.564 ± 0.105, 0.499 ± 0.343] (decision_tree)
        * F1 Macro = 0.496±0.153 (xgboost), F1 Micro = 0.534±0.137 (xgboost), F1 Weighted = 0.522±0.137 (xgboost), F1 class wise = [0.434 ± 0.262, 0.527 ± 0.085, 0.514 ± 0.281] (decision_tree)
        * AUC Macro = 0.689±0.156 (mlp), AUC Weighted = 0.698±0.144 (mlp)
    * Subject-wise metrics: 
        * Accuracy = 0.62±0.25 (xgboost)
        * Precision Macro = 0.6±0.226 (xgboost), Precision Micro = 0.62±0.25 (xgboost), Precision Weighted = 0.93±0.064 (decision_tree), Precision class wise = [0.6 ± 0.547, 0.7 ± 0.273, 0.5 ± 0.5] (decision_tree)
        * Recall Macro = 0.561±0.289 (decision_tree), Recall Micro = 0.62±0.25 (xgboost), Recall Weighted = 0.62±0.25 (xgboost), Recall class wise = [0.5 ± 0.5, 0.45 ± 0.389, 0.366 ±  0.414] (kernel_svm)
        * F1 Macro = 0.531±0.278 (xgboost), F1 Micro = 0.62±0.25 (xgboost), F1 Weighted = 0.682±0.161 (decision_tree), F1 class wise = [0.6 ± 0.547, 0.7 ± 0.273, 0.5 ± 0.5] (decision_tree)
        * AUC Macro = 0.833±0.139 (decision_tree), AUC Weighted = 0.849±0.113 (decision_tree)


* **Cross (task+subject) generalization (W -> WT)**
    * Stride-wise metrics: 
        * Accuracy = 0.482±0.075 (mlp)
        * Precision Macro = 0.493±0.084 (mlp), Precision Micro = 0.482±0.075 (mlp), Precision Weighted = 0.528±0.103 (mlp), Precision class wise = [0.436 ± 0.171, 0.407 ±0.099, 0.635 ± 0.241] (mlp)
        * Recall Macro = 0.495±0.078 (mlp), Recall Micro = 0.482±0.075 (mlp), Recall Weighted = 0.482±0.075 (mlp), Recall class wise = [0.615 ± 0.222, 0.475 ± 0.217, 0.393 ± 0.188] (mlp)
        * F1 Macro = 0.469±0.081 (mlp), F1 Micro = 0.482±0.075 (mlp), F1 Weighted = 0.476±0.078 (mlp), F1 class wise = [0.497 ± 0.181, 0.431 ± 0.142, 0.476 ± 0.202] (mlp)
        * AUC Macro = 0.676±0.06 (mlp), AUC Weighted = 0.681±0.049 (mlp)
    * Subject-wise metrics: 
        * Accuracy = 0.5±0.15 (decision_tree)
        * Precision Macro = 0.544±0.159 (decision_tree), Precision Micro = 0.5±0.15 (decision_tree), Precision Weighted = 0.781±0.178 (decision_tree), Precision class wise = [0.8 ± 0.273, 0.7 ± 0.447, 0.133 ± 0.182] (decision_tree)
        * Recall Macro = 0.5±0.165 (random_forest), Recall Micro = 0.5±0.15 (decision_tree), Recall Weighted = 0.5±0.15 (decision_tree), Recall class wise = [0.54 ± 0.456, 0.313 ± 0.301, 0.3 ± 0.447] (kernel_svm)
        * F1 Macro = 0.438±0.166 (mlp), F1 Micro = 0.5±0.15 (decision_tree), F1 Weighted = 0.572±0.163 (decision_tree), F1 class wise = [0.8 ± 0.273, 0.7 ± 0.447, 0.133 ±0.182] (decision_tree)
        * AUC Macro = 0.733±0.118 (mlp), AUC Weighted = 0.752±0.096 (mlp)

### Single stride traditional ML-based benchmark results for the Ablation Study
* **Task generalization (W -> WT)**
    * Feet (+speed): [Linear SVM]
        * Stride-wise metrics: 
            * Accuracy = 0.7303 
            * Precision Macro = 0.739, Precision Micro = 0.7303, Precision Weighted = 0.738, Precision class wise = [0.664 0.706 0.848]
            * Recall Macro = 0.729, Recall Micro = 0.730, Recall Weighted = 0.730, Recall class wise = [0.772 0.674  0.741]
            * F1 Macro = 0.732, F1 Micro = 0.730, F1 Weighted = 0.731, F1 class wise = [0.714  0.690 0.791]
            * AUC Macro = 0.871, AUC Weighted = 0.871
        * Subject-wise metrics: 
            * Accuracy = 0.92
            * Precision Macro = 0.933, Precision Micro = 0.92, Precision Weighted = 0.935, Precision class wise = [0.8 1.  1. ]
            * Recall Macro = 0.925, Recall Micro = 0.92, Recall Weighted = 0.92, Recall class wise = [1. 0.777 1. ]
            * F1 Macro = 0.921, F1 Micro = 0.92, F1 Weighted = 0.919, F1 class wise = [0.88888889 0.875 1. ]
            * AUC Macro = 0.988, AUC Weighted = 0.988

    * Feet + Ankle (+speed): [Linear SVM]
        * Stride-wise metrics: 
            * Accuracy = 0.720
            * Precision Macro = 0.727, Precision Micro = 0.720, Precision Weighted = 0.726, Precision class wise = [0.656 0.713 0.813]
            * Recall Macro = 0.719, Recall Micro = 0.720, Recall Weighted = 0.720, Recall class wise = [0.760 0.653 0.744]
            * F1 Macro = 0.721, F1 Micro = 0.720, F1 Weighted = 0.721, F1 class wise = [0.704 0.682 0.777]
            * AUC Macro = 0.871, AUC Weighted = 0.871
        * Subject-wise metrics: 
            * Accuracy = 0.92
            * Precision Macro = 0.933, Precision Micro = 0.92, Precision Weighted = 0.935, Precision class wise = [0.8 1.  1. ]
            * Recall Macro = 0.925, Recall Micro = 0.92, Recall Weighted = 0.92, Recall class wise = [1. 0.777 1. ]
            * F1 Macro = 0.921, F1 Micro = 0.92, F1 Weighted = 0.919, F1 class wise = [0.88888889 0.875 1. ]
            * AUC Macro = 0.995, AUC Weighted = 0.995

    * Feet + Ankle + Knee (+speed): [MLP]
        * Stride-wise metrics: 
            * Accuracy = 0.797
            * Precision Macro = 0.804, Precision Micro =  0.797, Precision Weighted = 0.803, Precision class wise = [0.748 0.761 0.903]
            * Recall Macro = 0.795, Recall Micro = 0.797, Recall Weighted =  0.797, Recall class wise = [0.866 0.731 0.789]
            * F1 Macro = 0.797, F1 Micro = 0.797, F1 Weighted = 0.797, F1 class wise = [0.803 0.746 0.842]
            * AUC Macro = 0.919, AUC Weighted = 0.919
        * Subject-wise metrics: 
            * Accuracy = 1.0
            * Precision Macro = 1, Precision Micro = 1, Precision Weighted = 1, Precision class wise = [1 1.  1. ]
            * Recall Macro = 1, Recall Micro = 1, Recall Weighted = 0.92, Recall class wise = [1. 1 1. ]
            * F1 Macro = 1, F1 Micro = 1, F1 Weighted = 1, F1 class wise = [1 1 1. ]
            * AUC Macro = 1, AUC Weighted = 1

* **Subject generalization (W)**
    * Feet (+speed): [MLP] 
        * Stride-wise metrics: 
            * Accuracy = 0.553$`\pm`$0.053 
            * Precision Macro = 0.544$`\pm`$0.069, Precision Micro = 0.553 $`\pm`$ 0.053, Precision Weighted = 0.589 $`\pm`$ 0.052, Precision class wise = [0.669$`\pm`$0.063, 0.472$`\pm`$0.202, 0.491$`\pm`$0.280] 
            * Recall Macro = 0.520$`\pm`$0.065, Recall Micro = 0.553 $`\pm`$ 0.053, Recall Weighted = 0.553 $`\pm`$ 0.053, Recall class wise = [0.650$`\pm`$0.147, 0.478$`\pm`$0.048, 0.432$`\pm`$0.211] 
            * F1 Macro = 0.518 $`\pm`$ 0.063, F1 Micro = 0.553 $`\pm`$ 0.053, F1 Weighted = 0.559 $`\pm`$ 0.047, F1 class wise = [0.657$`\pm`$0.100, 0.458$`\pm`$0.084, 0.440$`\pm`$0.222]  
            * AUC Macro = 0.697 $`\pm`$ 0.068, AUC Weighted = 0.707 $`\pm`$ 0.064
        * Subject-wise metrics: 
            * Accuracy = 0.752 $`\pm`$ 0.068
            * Precision Macro = 0.7 ± 0.09, Precision Micro = 0.75 ± 0.068, Precision Weighted = 0.826 ± 0.123, Precision class wise = [0.799 ± 0.182, 0.9 ± 0.223, 0.4 ± 0.418]
            * Recall Macro = 0.711 ± 0.213 (logistic_regression), Recall Micro = 0.752 ± 0.172 (logistic_regression), Recall Weighted = 0.752 ± 0.172 (logistic_regression), Recall class wise = [0.866 ± 0.182, 0.666 ± 0.204, 0.6 ± 0.547]
            * F1 Macro = 0.682 ± 0.09, F1 Micro = 0.752 ± 0.0683, F1 Weighted = 0.766 ± 0.0898, F1 class wise = [0.799 ± 0.182, 0.9 ± 0.223, 0.4 ± 0.418]
            * AUC Macro = 0.777 ± 0.092, AUC Weighted = 0.790 ± 0.0623

    * Feet + Ankle (+speed): [Decision Tree] 
        * Stride-based model performance (mean):  [0.5308044865591335, 0.5020094261038025, 0.5308044865591335, 0.5547174838887774, [0.6690790663976718, 0.4402477427960157, 0.39670146911772003], 0.5007365592026952, 0.5308044865591335, 0.5308044865591335, [0.6105530068317656, 0.5204722880124487, 0.37118438276387145], 0.4892721609319427, 0.5308044865591335, 0.5331537842763143, [0.6361570446654817, 0.46956101024244457, 0.3620984278879016], 0.6588744406916313, 0.6664810278219898]

        * Stride-based model performance (standard deviation):  [0.08340325838373583, 0.10616322782134671, 0.08340325838373583, 0.10830969659532365, [0.16633266916651343, 0.11755583157693998, 0.2509493896253871], 0.09813700017349407, 0.08340325838373583, 0.08340325838373583, [0.09054687457092121, 0.11407231032651302, 0.24865347029065432], 0.09591849353937755, 0.08340325838373583, 0.09127196770283343, [0.12345181612842324, 0.09173537830805983, 0.20986499478675932], 0.06945280210305836, 0.07023544039156958]

        * Person-based model performance (mean):  [0.6238095238095239, 0.611111111111111, 0.6238095238095239, 0.6809523809523809, [0.6333333333333333, 0.7, 0.5], 0.6555555555555556, 0.6238095238095239, 0.6238095238095239, [0.7, 0.6666666666666666, 0.6], 0.5980952380952382, 0.6238095238095239, 0.620952380952381, [0.6333333333333333, 0.7, 0.5], 0.6652777777777777, 0.6775793650793651]

        * Person-based model performance (standard deviation):  [0.13075743064519985, 0.17568209223157663, 0.13075743064519985, 0.15236441709227247, [0.07453559924999298, 0.27386127875258304, 0.3535533905932738], 0.18392161508052057, 0.13075743064519985, 0.13075743064519985, [0.29814239699997197, 0.31180478223116176, 0.4183300132670378], 0.15951993993569402, 0.13075743064519985, 0.1249898962910808, [0.07453559924999298, 0.27386127875258304, 0.3535533905932738], 0.17338318371474706, 0.15833780874448317]





