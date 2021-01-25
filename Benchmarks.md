# Gait Video Study

### Single stride traditional ML-based benchmark results
* Task generalization (W -> WT)
    * Stride-wise metrics: Accuracy = 0.787 (GBM), Precision = 0.784 (GBM), Recall = 0.779 (RBF SVM), F1 = 0.750 (XGBoost), AUC = 0.867 (GBM/XGBoost)
    * Subject-wise metrics: Accuracy = 0.943 (GBM), Precision = 1.0 (GBM), Recall = 0.882 (GBM), F1 = 0.938 (GBM), AUC = 1.0 (GBM)
<br>
* Subject generalization (W)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)
<br>
* Subject generalization (WT)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)
<br>
* Subject generalization (comparing W and WT: W)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)
<br>
* Subject generalization (comparing W and WT: WT)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)
<br>
* Cross (task+subject) generalization (W -> WT)
    * Stride-wise metrics: Accuracy = 0.592$`\pm`$0.15 (AdaBoost), Precision = 0.595$`\pm`$0.23 (AdaBoost), Recall = $`0.459\pm0.20`$ (Decision Tree), F1 = $`0.451\pm0.19`$ (AdaBoost), AUC = $`0.644\pm0.18`$ (AdaBoost)
    * Subject-wise metrics: Accuracy = $`0.571\pm0.20`$ (MLP), Precision = $`0.548\pm0.34`$ (RF), Recall = $`0.548\pm0.33`$ (RF), F1 = $`0.514\pm0.29`$ (RF), AUC = $`0.774\pm0.15`$ (AdaBoost)
<br>
