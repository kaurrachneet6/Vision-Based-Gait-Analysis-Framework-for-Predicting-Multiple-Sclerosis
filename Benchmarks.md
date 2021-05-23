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

    * Feet + Ankle + Knee (+speed): [Decision Tree] 

        * Stride-based model performance (mean):  [0.552431565485874, 0.523475025581894, 0.552431565485874, 0.5627743879577961, [0.6456621872244801, 0.4406732874968169, 0.48408960202438467], 0.5114195769724093, 0.552431565485874, 0.552431565485874, [0.7050306489822208, 0.38473772583210336, 0.4444903561029038], 0.5048833541699165, 0.552431565485874, 0.5476031049196527, [0.6697295539485032, 0.4063994808063441, 0.4385210277549021], 0.657367547028216, 0.6657754245372374]

        * Stride-based model performance (standard deviation):  [0.07326112444199018, 0.06406263614442786, 0.07326112444199018, 0.08638317692093374, [0.13366756692826245, 0.18461571660067402, 0.1332704863641986], 0.07779315044193519, 0.07326112444199018, 0.07326112444199018, [0.13387948749819273, 0.10126691752818218, 0.21017746307034926], 0.06089264672761367, 0.07326112444199018, 0.07331682185168384, [0.11454386292698084, 0.13253688090187568, 0.10136957218287436], 0.07408900315646101, 0.07609228148939128]

        * Person-based model performance (mean):  [0.6285714285714284, 0.5888888888888888, 0.6285714285714284, 0.7992063492063493, [0.8666666666666667, 0.4, 0.5], 0.58, 0.6285714285714284, 0.6285714285714284, [0.6733333333333333, 0.4666666666666667, 0.6], 0.5466666666666666, 0.6285714285714284, 0.6705555555555556, [0.8666666666666667, 0.4, 0.5], 0.8111111111111111, 0.813095238095238]

        * Person-based model performance (standard deviation):  [0.11527350892295433, 0.1515353521887317, 0.11527350892295433, 0.12253104665679543, [0.18257418583505539, 0.4183300132670378, 0.3535533905932738], 0.16546231527987812, 0.11527350892295433, 0.11527350892295433, [0.19206480387850577, 0.5055250296034367, 0.4183300132670378], 0.15025081910583823, 0.11527350892295433, 0.09629911618639958, [0.18257418583505539, 0.4183300132670378, 0.3535533905932738], 0.11869922566234162, 0.11533496518840824]


* **Subject generalization (WT)**
    * Feet (+speed): [RBF SVM] 
        * Stride-based model performance (mean):  [0.4939371468492837, 0.41175278587271286, 0.4939371468492837, 0.4360032121669552, [0.416903881492982, 0.10420332355816227, 0.7141511525669941], 0.49292746688170785, 0.4939371468492837, 0.4939371468492837, [0.6694531430889156, 0.11854103343465044, 0.6907882241215575], 0.4050844819963828, 0.4939371468492837, 0.4177276985228072, [0.5050489948823194, 0.10505606092659192, 0.6051483901802373], 0.7090426046827972, 0.7176777500046387]

        * Stride-based model performance (standard deviation):  [0.12158635058147987, 0.16792850767766596, 0.12158635058147987, 0.18549885273612368, [0.2743285277102329, 0.18639023867104199, 0.25173279571451174], 0.10199833530608475, 0.12158635058147987, 0.12158635058147987, [0.37541312369584884, 0.25333940807373706, 0.33334456140806873], 0.12037741161721909, 0.12158635058147987, 0.13172343709589787, [0.30845963557564926, 0.21615238382140273, 0.12859523650858667], 0.0993975744309923, 0.08551706366976419]

        * Person-based model performance (mean):  [0.5166666666666666, 0.5333333333333333, 0.5166666666666666, 0.8416666666666668, [0.8, 0.1, 0.7], 0.4111111111111111, 0.5166666666666666, 0.5166666666666666, [0.4333333333333333, 0.1, 0.7], 0.42666666666666664, 0.5166666666666666, 0.6011111111111112, [0.8, 0.1, 0.7], 0.6583333333333333, 0.66875]

        * Person-based model performance (standard deviation):  [0.15275252316519466, 0.12472191289246472, 0.15275252316519466, 0.0927960727138337, [0.4472135954999579, 0.223606797749979, 0.27386127875258304], 0.19751543149590198, 0.15275252316519466, 0.15275252316519466, [0.2788866755113585, 0.223606797749979, 0.29814239699997197], 0.16384274303259344, 0.15275252316519466, 0.14155878245057388, [0.4472135954999579, 0.223606797749979, 0.27386127875258304], 0.1674979270186815, 0.15314979957907585]


    * Feet + Ankle (+speed): [RBF SVM] 
        * Stride-based model performance (mean):  [0.4879350878238891, 0.4469492912152496, 0.4879350878238891, 0.4683777877115446, [0.4277393831588565, 0.19211674849972724, 0.7209917419871653], 0.49256696337148276, 0.4879350878238891, 0.4879350878238891, [0.5929002229443876, 0.19382250952523652, 0.6909781576448243], 0.41679225168503453, 0.4879350878238891, 0.4261520892926244, [0.4786618459494121, 0.15970399933482995, 0.6120109097708618], 0.7051583395634975, 0.7143491276070163]

        * Stride-based model performance (standard deviation):  [0.12020399722114787, 0.17114674152841128, 0.12020399722114787, 0.1882284611198244, [0.27994245207223434, 0.20303377125043298, 0.2495301376728465], 0.10158609236588112, 0.12020399722114787, 0.12020399722114787, [0.36283506907662, 0.335841934825919, 0.32699581116456167], 0.12255500816193782, 0.12020399722114787, 0.13352356109810437, [0.2938140096219309, 0.21967065849818304, 0.11961461376721552], 0.10685737147150719, 0.09346203504095332]

        * Person-based model performance (mean):  [0.5166666666666666, 0.5333333333333333, 0.5166666666666666, 0.9083333333333332, [0.7, 0.2, 0.7], 0.3822222222222222, 0.5166666666666666, 0.5166666666666666, [0.5333333333333333, 0.08, 0.5333333333333333], 0.40698412698412695, 0.5166666666666666, 0.6207936507936509, [0.7, 0.2, 0.7], 0.675, 0.6854166666666666]

        * Person-based model performance (standard deviation):  [0.15275252316519466, 0.1247219128924647, 0.15275252316519466, 0.0927960727138337, [0.4472135954999579, 0.447213595499958, 0.4472135954999579], 0.1358285047994322, 0.15275252316519466, 0.15275252316519466, [0.380058475033046, 0.17888543819998318, 0.380058475033046], 0.13419449789505575, 0.15275252316519466, 0.16249747560882025, [0.4472135954999579, 0.447213595499958, 0.4472135954999579], 0.181429508808977, 0.16723860200324558]

    * Feet + Ankle + Knee (+speed): [MLP] 
        * Stride-based model performance (mean):  [0.49472938630378127, 0.5093303997919442, 0.49472938630378127, 0.5401356787490561, [0.3644885256577929, 0.5077677014947517, 0.6557349722232881], 0.4945139225557532, 0.49472938630378127, 0.49472938630378127, [0.38988183892719575, 0.438537917096313, 0.6551220116437508], 0.46862916117067577, 0.49472938630378127, 0.4791051969979711, [0.3728277666713852, 0.4401585278533476, 0.5929011889872946], 0.6335675901017128, 0.6386209023373931]

        * Stride-based model performance (standard deviation):  [0.13611555443961, 0.09271183205261595, 0.13611555443961, 0.06428239637588327, [0.3206709892124471, 0.28247675374685977, 0.12542480712030674], 0.1280561055757218, 0.13611555443961, 0.13611555443961, [0.34676196449156443, 0.16787402231734583, 0.3153657097018935], 0.13964778147259285, 0.13611555443961, 0.1364763456882792, [0.3249116301786096, 0.15918109366918043, 0.19336992824046034], 0.1472443609728228, 0.14321266634021393]

        * Person-based model performance (mean):  [0.5833333333333333, 0.6, 0.5833333333333333, 0.9083333333333332, [0.6, 0.5, 0.7], 0.4666666666666666, 0.5833333333333333, 0.5833333333333333, [0.4666666666666667, 0.39999999999999997, 0.5333333333333333], 0.4888888888888888, 0.5833333333333333, 0.6722222222222223, [0.6, 0.5, 0.7], 0.7125, 0.7354166666666666]

        * Person-based model performance (standard deviation):  [0.19720265943665388, 0.16996731711975951, 0.19720265943665388, 0.09279607271383371, [0.5477225575051661, 0.5, 0.4472135954999579], 0.24745619390355653, 0.19720265943665388, 0.19720265943665388, [0.4472135954999579, 0.4346134936801766, 0.380058475033046], 0.2139574204784162, 0.19720265943665388, 0.18313962830799388, [0.5477225575051661, 0.5, 0.4472135954999579], 0.22376202636829254, 0.19847508939690922]


* **Cross generalization (W->WT)**
    * Feet (+speed): [MLP] 

    * Feet + Ankle (+speed): [Decision Tree] 

    * Feet + Ankle + Knee (+speed): [Decision Tree]     



