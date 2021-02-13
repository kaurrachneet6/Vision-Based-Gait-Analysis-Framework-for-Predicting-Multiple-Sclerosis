# Gait Video Study

### Single stride traditional ML-based benchmark results
* Task generalization (W -> WT)
    * Stride-wise metrics: 
        * Accuracy = 0.8307 (XGBoost)
        * Precision Macro = 0.835 (XGBoost), Precision Micro = 0.8307 (XGBoost), Precision Weighted = 0.834 (XGBoost), Precision class wise = [0.774, 0.832, 0.899] (XGBoost)
        * Recall Macro = 0.829 (XGBoost), Recall Micro = 0.8307 (XGBoost), Recall Weighted = 0.8307 (XGBoost), Recall class wise = [0.888, 0.716, 0.882] (XGBoost)
        * F1 Macro = 0.829 (XGBoost), F1 Micro = 0.8307 (XGBoost), F1 Weighted = 0.829 (XGBoost), F1 class wise = [0.827, 0.770, 0.890] (XGBoost)
        * AUC Macro = 0.945 (GBM), AUC Weighted = 0.945 (GBM)
    * Subject-wise metrics: 
        * Accuracy = 
        * Precision Macro = , Precision Micro = , Precision Weighted = , Precision class wise =
        * Recall Macro = , Recall Micro = , Recall Weighted = , Recall class wise = 
        * F1 Macro = , F1 Micro = , F1 Weighted = ), F1 class wise = 
        * AUC Macro = , AUC Weighted = 


* Subject generalization (W)
    * Stride-wise metrics: 
        * Accuracy = 0.592$`\pm`$0.15 (AdaBoost)
        * Precision = 0.595$`\pm`$0.23 (AdaBoost)
        * Recall = $`0.459\pm0.20`$ (Decision Tree)
        * F1 = $`0.451\pm0.19`$ (AdaBoost)
        * AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)


* Subject generalization (WT)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)


* Subject generalization (comparing W and WT: W)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)


* Subject generalization (comparing W and WT: WT)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)


* Cross (task+subject) generalization (W -> WT)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)


