from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *
from ml_utils.positional_encoding import PositionalEncoder

'''
Conv1D reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
'''

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, dilation, groups, batch_norm, dropout, maxpool, maxpool_kernel_size):
        '''
        in_chans: input channels/features
        out_chans: output channels/features for the 1d conv layer
        kernel_size
        stride: default = 1
        dilation: default = 1, dilation controls the spacing between the kernel points
        groups: default = 1, groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups
        batch_norm: True or False to use it for the conv block
        dropout: 0 means no dropout else 0<p<1 means during training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution
        maxpool: True or False to use it for the conv block
        maxpool_kernel_size: kernel size used for the maxpool layer if maxpool is True
        Conv Block: Conv1D - Batchnorm1D - ReLu - Dropout - Maxpool1D
        '''
        super(ConvBlock, self).__init__()
        layers = []
        self.conv1 = nn.Conv1d(in_chans, out_chans, kernel_size = kernel_size, stride = stride, dilation = dilation, groups = groups)
        layers += [self.conv1]
        if batch_norm:
            self.batch_norm_layer =  nn.BatchNorm1d(num_features=out_chans)
            layers += [self.batch_norm_layer]
        self.relu = nn.ReLU()
        layers += [self.relu]
        self.dropout_layer = nn.Dropout(dropout)
        layers += [self.dropout_layer]
        if maxpool:
            self.maxpool_layer = nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            layers += [self.maxpool_layer]
        self.convblock_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.convblock_net(x)
    

    
class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        in_dim: input dimension for the dense layer 
        out_dim: output dimension for the dense layer
        DenseBlock: Dense - ReLU
        '''
        super(DenseBlock, self).__init__()
        self.denseblock_net = nn.Sequential(nn.Linear(in_dim, out_dim), 
                                          nn.ReLU())
    def forward(self, x):
        return self.denseblock_net(x)
    
    
class CNN1D(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, dilation, groups, batch_norm, dropout, maxpool, maxpool_kernel_size, dense_out_sizes, dense_pool, dense_pool_kernel_size, dense_dropout, global_average_pool, num_classes, time_steps, position_encoding):
        '''
        num_conv_block_layers = len(out_chans)
        Single conv Block: Conv1D - Batchnorm1D - ReLu - Dropout - Maxpool1D
        in_chans = 36
        out_chans = [64, 128, 256, ..., 64], where each element represents out_chans for that layer's conv. block
        kernel_size = [8, 5, 3, ..., 5]
        stride = [2, 1, ..., 1]
        dilation = [1, 1, ...]
        groups = [1, ...]
        batch_norm = [True, False, ...]
        dropout = [0.3, ...]
        maxpool = [True, False, ...]
        maxpool_kernel_size: [2, 3, ...] = kernel size used for the maxpool layer if maxpool is True
        
        dense_out_sizes: 
            [50, 10] means we will have 3 dense layers after the CNN layers, first, from flattened CNN 'out channels' to 50 dimensions, second, from 50 to 10 dimensions, and last, from 10 dimensions to 3 output class probabilities.
            [] means we will have only 1 dense layer after the CNN layers, from the flattened CNN 'out channels' dimensions to 3 output class probabilities.
            [10] means we will have 2 dense layers, first, from CNN 'out channels' to 10 dimensions and then second, from 10 to 3.
            len(dense_out_sizes)+1 is the number of dense layers we would have 
        
        dense_pool: True or False depending on whether we would have maxpool after the CNN layers or not 
        dense_pool_kernel_size: If we do have maxpool after the CNN layers, the kernel size for that maxpool layer 
        dense_dropout: dropout probability between the dense layers [0.3, 0.5] for 2 in between dense layers 
        
        global_average_pool: True or False, depending on whether we would need a global average pooling after the CNN blocks to replace the dense layers or not. 
        
        For details on Global Average Pooling, refer https://paperswithcode.com/method/global-average-pooling
        Global Average Pooling is a pooling operation designed to replace fully connected layers in classical CNNs. The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer. Instead of adding fully connected layers on top of the feature maps, we take the average of each feature map, and the resulting vector is fed directly into the softmax layer.
One advantage of global average pooling over the fully connected layers is that it is more native to the convolution structure by enforcing correspondences between feature maps and categories. Thus the feature maps can be easily interpreted as categories confidence maps. Another advantage is that there is no parameter to optimize in the global average pooling thus overfitting is avoided at this layer. Furthermore, global average pooling sums out the spatial information, thus it is more robust to spatial translations of the input.
        num_classes = 3
        time_steps = 20
        
        CNN1D: n*conv_blocks -> pooling -> dense layers (one or a few)
        '''
        super(CNN1D, self).__init__()
        self.global_average_pool = global_average_pool
        self.position_encoding = position_encoding
        #Positional encoding to the input features 
        if self.position_encoding:
            self.pos_encoding_layer = PositionalEncoder(in_chans)
        cnn_layers = []
        num_layers = len(out_chans)
        for i in range(num_layers):
            in_channels = in_chans if i == 0 else out_chans[i-1]
            cnn_layers += [ConvBlock(in_channels, out_chans[i], kernel_size[i], stride[i], dilation[i], groups[i], batch_norm[i], dropout[i], maxpool[i], maxpool_kernel_size[i])]
        
        #If there is maxpool layer after the CNN blocks, we execute it here 
        if dense_pool:
            self.dense_maxpool_layer = nn.MaxPool1d(kernel_size=dense_pool_kernel_size)
            cnn_layers += [self.dense_maxpool_layer]
        
        #Running through the CNN blocks 
        self.cnn_network = nn.Sequential(*cnn_layers)
        
        #To get the input dimensions for the first dense layer after the CNN network, we pass a random tensor of same 
        #shape as input shape for our data samples and then use the number of output features as the cnn_out_features 
        random_tensor = torch.rand(1, in_chans, time_steps)
        random_tensor_out = self.cnn_network(random_tensor)
        cnn_out_features = random_tensor_out.shape[2]
        
        if self.global_average_pool:
            dense_out_dim = out_chans[-1] + 1 #+1 is for the extra frame count feature added to the flattened vector      
        else:
            #Since we flatten the output of the CNN network for the dense layers, the input to the first dense layer 
            #is shaped output_cnn_channels*output_cnn_features 
            #+1 is for the extra frame count feature added to the flattened vector 
            dense_out_dim = out_chans[-1]*cnn_out_features + 1 #Out put channels from the last conv block
        
        dense_layers = []
        num_dense_layers = len(dense_out_sizes) 
        
        for i in range(num_dense_layers):
            #+1 is for the extra frame count feature added to the flattened vector 
            dense_in_dim = out_chans[-1]*cnn_out_features + 1 if ((i == 0) and not(self.global_average_pool)) else out_chans[-1] + 1 if ((i == 0) and (self.global_average_pool)) else dense_out_sizes[i-1]
            dense_out_dim = dense_out_sizes[i]
            dense_layers += [DenseBlock(dense_in_dim, dense_out_dim)] #Dense + ReLU
            dense_layers += [nn.Dropout(dense_dropout[i])] #Added dropout if p>0
        
        dense_layers += [nn.Linear(dense_out_dim, num_classes)] #Last mandatory linear layer for the 3 output probabilities
        self.dense_network = nn.Sequential(*dense_layers)
        
    
    def forward(self, body_coords, frame_count):               
        #Positional encoding 
        if self.position_encoding:
            body_coords = self.pos_encoding_layer(body_coords)
        
        # At input, boody coords is shaped as batch_size x sequence length (time steps) x 36 features 
        #But for 1D CNN, the input should be shaped as batch_size x features/channels x time steps/sequence length
        body_coords = body_coords.permute(0,2,1)
        
        #Running through the CNN blocks (Conv1D - Batchnorm - ReLU - Maxpool - Dropout) + an additional maxpool after all the 
        #CNN blocks, if needed
        x = self.cnn_network(body_coords)
        #After running the body coordinate inputs via a CNN network, we either apply the Global Average Pooling 
        # to reducing the dimensions to batch_size*features 
        if self.global_average_pool:
            x = x.mean(2)
       #or otherwise flatten the CNN output to batch_size*features 
        else:
            #Flattening for the dense layers 
            x = x.view(-1, x.shape[1]*x.shape[2])
        #Concatenating the frame count as an additional feature 
        x = torch.cat((x, frame_count.unsqueeze(dim = 1)), dim = 1).float() 
        #After concatenating the frame count as an additional feature to our flattened vector, we pass through the 
        #dense network (Dense layer - ReLU - Dropout) + Last mandatory linear layer for the 3 output probabilities
        x = self.dense_network(x)         
        return x