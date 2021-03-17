from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *

#Pytorch dataset definition
class GaitDataset(Dataset_skorch):
    #We need to add the frame count as an extra feature along with 36 features for each stride 
    def __init__(self, data_path, labels_csv, pids_retain, framework = 'W', datastream = 'All', train_frame_count_mean = None, train_frame_count_std = None):   
        '''
        Arguments: 
            data_path: data path for downsampled strides 
            labels_csv: csv file with labels 
            pids_retain: PIDs to return data for 
            framework: Task to return data for 
            transforms: For ToTensor transformation of dataframes
            train_frame_count_mean: Mean for the training data frame count (computed beforehand) to z-score normalize the training and testing samples 
            train_frame_count_std: Standard deviation for the training data frame count (computed beforehand) to z-score normalize the training and testing samples
        
        Returns:
            data: data = {'body_coords': X, 'frame_count': frame_count}
                A dictionary with X (20 rows for 20 downsampled frames per stride and 36 columns for 36 body coordinate features 
                for each sample) and frame_count (the original count of frames per stride before downsampling). The X features are z-score 
                normalized within the stride and frame_count is z-score normalized across the training data frame count and converted to tensor.
            y: PID and label for each sample. These values are converted to tensor.
        '''
        #Assigning the data folder for the downsampled strides 
        self.data_path = data_path
        #Reading the labels file
        self.all_labels = pd.read_csv(labels_csv, index_col = 0)
        self.datastream = datastream
        #Retaining only the labels dataframe for framework and PIDs of interest and resetting the index
        if type(framework) is str:
            self.reduced_labels = self.all_labels[self.all_labels.scenario == framework][self.all_labels.PID.isin(pids_retain)].reset_index()
        else: #List type 
            self.reduced_labels = self.all_labels[self.all_labels.scenario.isin(framework)][self.all_labels.PID.isin(pids_retain)].reset_index()
        #Setting the labels with index as the key and PID along with to use when computing subject wise evaluation metrics
        self.labels = self.reduced_labels[['PID', 'label', 'key']].set_index('key')
        self.len = len(self.labels) #Length of the data to use
        self.transforms = transforms
        self.train_frame_count_mean = train_frame_count_mean
        self.train_frame_count_std = train_frame_count_std
        self.epsilon = 10**(-6)
        self.__define_datastreams__()
    
    def __len__(self):
        #Returns the length of the data 
        return self.len
    
    def __define_datastreams__(self):
        '''
        Used for Ablation study on body coordinates 
        '''
        random_key = self.reduced_labels['key'].iloc[0]
        random_X = pd.read_csv(self.data_path+random_key+'.csv', index_col = 0)
        self.feet_features = [s for s in random_X.columns if any(x in s for x in ['toe', 'heel'])]
        ankle_features = [s for s in random_X.columns if 'ankle' in s]
        knee_features = [s for s in random_X.columns if 'knee' in s]
        self.feet_ankle_features = self.feet_features + ankle_features
        self.feet_ankle_knee_features = self.feet_features + ankle_features + knee_features
          

    def __getitem__(self, index):
        #Generates one sample of data
        #Select key to sample
        key = self.reduced_labels['key'].iloc[index]

        # Load data and get label
        X = pd.read_csv(self.data_path+key+'.csv', index_col = 0)
        X = (X-X.mean())/(X.std()+self.epsilon) #Within stride z-score normalization for all 36 coordinate features 
        #Creating a new frame count column represting the total original count of frames in a stride 
        #denoting the speed of the stride
        frame_count = float(self.reduced_labels[self.reduced_labels['key']==key]['frame_count'].values[0])
        y = self.labels.loc[key] #PID and label extracted for the key at the index 
        #X- 20 rows for 20 downsampled frames per stride and 37 columns for 37 features for each sample
        #y - PID and label for each sample
        if self.train_frame_count_mean is not None: #Used to load z-score normalized data in batches 
            #Across training data strides, frame count normalization
            frame_count = (frame_count - self.train_frame_count_mean)/self.train_frame_count_std 
        
        if self.datastream == 'feet':
            X = X[self.feet_features]
        if self.datastream == 'feet_ankle':
            X = X[self.feet_ankle_features]
        if self.datastream == 'feet_ankle_knee':
            X = X[self.feet_ankle_knee_features]

        X = torch.Tensor(X.values) #converting the dataframe to tensor 
        data = {'body_coords': X, 'frame_count': frame_count}
        label = torch.Tensor(y)[1:].long() #shape = 2 for PID and label 
#         print (label.shape)
        pid = torch.Tensor(y)[:1]
#         print (X.shape, y.shape)
        return data, label.squeeze(), pid   
