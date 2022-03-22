#### Utility functions 
* **imports.py**: package imports
* **split.py**: Function definition for StratifiedGroupKFold for handling imbalance in case of Group K fold https://github.com/scikit-learn/scikit-learn/issues/13621
* **task_gen_traditionalML.py**: Functions to read extracted features data, tune and evaluate traditional ML models, plot confusion matrices and ROC curves for tuned models in task generalization frameworks.
* **subject_gen_traditionalML.py**: Functions to read extracted features data, tune and evaluate traditional ML models, plot confusion matrices and ROC curves for tuned models in subject generalization frameworks.
* **cross_gen_traditionalML.py**: Functions to read extracted features data, tune and evaluate traditional ML models, plot confusion matrices and ROC curves for tuned models in cross (task + subject) generalization frameworks.
* **gait_data_loader.py**: Defines the Data loader for the deep learning frameworks 
* **DLutils.py**: Contains definition of general utilities like setting random seed for replicability etc. used across all three generalization frameworks and deep learning models 
* **task_gen_DLtrainer.py**: Utility functions like train, resume train, evaluate etc. for training the deep learning models on the task generalization framework
* **subject_gen_DLtrainer.py**: Utility functions for training the deep learning models on the subject generalization frameworks
* **cross_gen_DLtrainer.py**: Utility functions for training the deep learning models on the cross generalization frameworks
* **cnn1d_model.py**: CNN1D model for time series classification with and without positional encoding
* **positional_encoding.py**: Positional encoding https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
* **RESNET_model.py**: Residual 1D model for time series classification with and without positional encoding
* **padding.py**: Implementation for "padding = same" in Pytorch https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/padding.py#L28
* **MULTISCALE_RESNET_model.py**: Multiscale Residual network for time series classification https://github.com/geekfeiw/Multi-Scale-1D-ResNet
* **TCN_model.py**: Temporal Convolutional Model 
* **RNN_model.py**: Vanilla Recurrent Neural Network (Uni- and Bi-directional versions)
* **GRU_model.py**: Gated Recurrent Unit model (Uni- and Bi-directional)
* **LSTM_model.py**: Long-short term memory model (Uni- and Bi-directional)
