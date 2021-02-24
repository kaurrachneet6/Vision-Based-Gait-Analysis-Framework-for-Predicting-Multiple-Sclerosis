from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *

'''
WeightNorm reference: https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f
'''
class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
    

'''
Temporal Convolutional Model (TCN) reference: https://github.com/locuslab/TCN
'''

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    '''
    Defines a single TCN block with 1D dialated fully convolutional layer -> Weight norm -> ReLU -> Dropout -> 1D dialated fully convolutional layer -> Weight norm -> ReLU -> Dropout
    The length of the output sequence is exactly same as the length of the input sequence (as in LSTMs).
    The stride is 1 and padding size is (kernet size -1)*dialiation size to ensure the fully conv. layer setup and hence making sure the output length of the block is same as the input length.
    The dialation size depends on the layer in the TCN network. For layer i, the dialation size for the temporal block is 2^i.
    n_inputs = channels/features in the input of this TCN block
    n_outputs = output channels/hidden size of this TCN block 
    '''
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        '''
        Input size corresponds to [batch_size, in_channels, len]. Each kernel in your conv layer creates an output channel, and convolves the “temporal dimension”, i.e. the len dimension.
        '''
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.conv2 =  nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        self.conv1 =  WeightNorm(self.conv1, weights = ['weight'])
        self.conv2 = WeightNorm(self.conv2, weights = ['weight'])
            
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)        
        self.relu = nn.ReLU()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    '''
    TCN network with multiple layers/TCN blocks 
    At 'i'th layer/TCN block of the TCN network, the dialation size is 2^i
    num_inputs = no. of channels/features in the input of the TCN model (in our case, we have 36 input features)
    num_channels = [hidden size/channels/features in layer 1 of TCN network, hidden size/channels/features in layer 2 of TCN network, ..., hidden size/channels/features in layer 'n' of TCN network]
    So, len(num_channels) = no. of layers/TCN blocks in the TCN network
    And no. of channels in each TCN block/layer 'i' is given by 'num_channels[i]'
    Dialation at layer/TCN block 'i' is 2^i
    This TCN network returns the output of exactly the same length as the input
    In input is N*C*L dimensional, where N is batch size, C is no. of input features/channels in each sample, L is the temporal dimension of the sample i.e. 20 frames per stride sample; then the output of this TCN network is N*H_n*L, where H_n is the number of channels/features in the last TCN block/layer. Note that the length of the output is same as the input length L.
    '''
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class TCN(nn.Module):
    '''
    This is the final TCN model for our many to one model/classification problem.
    This uses TCN network defined above (which in turn used the TCN block defined) and takes the output of the TCN network [N, H_n, L] dimesional at the last time step [N, H_n, -1] (as we do with LSTM also, when we have to map a sequence to a single label) and passes through a linear layer that maps H_n->out_channels (3 in our case for 3 classes) to get the final output in [N, out_channel] dimensions. 
    Note that since we have an extra speed denoting 'frame_count' feature before passing through the linear layer, we have H_n+1 input features to the linear layer and 3 output probabilities.
    input_size = in channels/features 
    output_size = out channels/features
    num_channels = [hidden size/channels/features in layer 1 of TCN network, hidden size/channels/features in layer 2 of TCN network, ..., hidden size/channels/features in layer 'n' of TCN network]
    So, len(num_channels) = no. of layers/TCN blocks in the TCN network
    And no. of channels in each TCN block/layer 'i' is given by 'num_channels[i]'
    kernel_size
    dropout
    '''
    def __init__(self, in_chans, out_chans, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(in_chans, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]+1, out_chans) #+1 for extra frame count feature 


    def forward(self, body_coords, frame_count):
        """
        Inputs have to have dimension (N, C_in, L_in), where N = batch size, C_in = input channels/features and L_in = number of time steps/temporal dimension of the sample
        For our case, it could be (100, 36, 20) dimensional since each stride has 20 frames and each frame has 36 body coordinate features 
        """
        # At input, boody coords is shaped as batch_size x sequence length (time steps) x 36 features 
        #But for 1D CNN, and thus TCN, the input should be shaped as batch_size x features/channels x time steps/sequence length
        body_coords = body_coords.permute(0,2,1)
        y1 = self.tcn(body_coords)  # input should have dimension (N, C, L)
        y = y1[:, :, -1] #Taking the output of the TCN network [N, H_n, L] dimesional at the last time step [N, H_n, -1]
        y = torch.cat((y, frame_count.unsqueeze(dim = 1)), dim = 1).float() #Concatenating the frame count as an additional feature 
        #Note that we do not use activation at the last layer for classification problems 
        out = self.linear(y) #Softmax happens at the cross entropy loss 
        return out
    

