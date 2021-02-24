from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *

class LSTM(nn.Module):
    '''
    Pytorch LSTM model class
    Functions:
        init: initializes model based on given parameters
        forward: forward step through model
    '''
    
    def __init__(self, in_chans, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional, pre_out, single_layer, linear_size, use_layernorm, batch_size, device1):
        super(LSTM, self).__init__()
        self.in_chans = in_chans        
        self.hidden_size1 = hidden_size1
        self.num_layers1 = num_layers1
        self.hidden_size2 = hidden_size2
        self.num_layers2 = num_layers2
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pre_out = pre_out
        self.single_lstm = single_layer
        self.linear_size = linear_size
        self.use_layernorm = use_layernorm 
        self.batch_size = batch_size
        self.device = device1  
        
        if (self.bidirectional):
            self.bi_mult = 2
        else:
            self.bi_mult = 1      
        
        if (self.single_lstm):
            hidden_size = self.hidden_size1
        else:
            hidden_size = self.hidden_size2

        self.h0, self.c0, self.h02, self.c02 = self.init_hidden()
        self.prefc = nn.Linear(self.in_chans, self.pre_out)

        #Batch_first means shape is (batch, seq, feature)
        self.lstm = nn.LSTM(self.in_chans, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional = self.bidirectional)
        self.lstm2 = nn.LSTM(self.hidden_size1*self.bi_mult, self.hidden_size2, self.num_layers2, batch_first=True, bidirectional = self.bidirectional) ##Use these two for two seperate hidden unit lstm layers

        self.dropout_layer = nn.Dropout(p = self.dropout)

        #Two fully connected layers after the LSTM layers
        self.fc0 = nn.Linear(hidden_size*self.bi_mult + 1, self.linear_size)  #+1 for extra frame count feature 
        self.fc1 = nn.Linear(self.linear_size, self.num_classes)

        #A single fully connected layer after the LSTM layers 
        self.fc = nn.Linear(hidden_size*self.bi_mult + 1, self.num_classes) #+1 for extra frame count feature 
        
        #Layernorm if needed
        self.layernorm = nn.LayerNorm(hidden_size*self.bi_mult)
    
    
    def init_hidden(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        #Xavier Normal Initialization 
        h = torch.empty(self.num_layers1*self.bi_mult, batch_size, self.hidden_size1)
        c = torch.empty(self.num_layers1*self.bi_mult, batch_size, self.hidden_size1)
        
        h0 = torch.nn.init.xavier_normal_(h, gain=1.0).to(self.device)
        c0 = torch.nn.init.xavier_normal_(c, gain=1.0).to(self.device)
        
        #h0 = torch.nn.init.kaiming_uniform_(h, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c0 = torch.nn.init.kaiming_uniform_(c, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        h2 = torch.empty(self.num_layers2*self.bi_mult, batch_size, self.hidden_size2)
        c2 = torch.empty(self.num_layers2*self.bi_mult, batch_size, self.hidden_size2)
        
        h02 = torch.nn.init.xavier_normal_(h2, gain=1.0).to(self.device)
        c02 = torch.nn.init.xavier_normal_(c2, gain=1.0).to(self.device)
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        return h0, c0, h02, c02



    def forward(self, body_coords, frame_count):
        #Initialize initial LSTM states
        h0 = self.h0
        c0 = self.c0
        h02 = self.h02
        c02 = self.c02
        if body_coords.size(0) != self.batch_size:
            h0, c0, h02, c02 = self.init_hidden(body_coords.size(0))
    
        #out = F.relu(self.prefc(body_coords)) #Uncomment for first linear layer

        # Forward propagate LSTM
        #out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size) #uncomment for first linear layer
        out, _ = self.lstm(body_coords, (h0, c0))    #uncomment for first lstm layer
        if (not self.single_lstm):
            out, _ = self.lstm2(out, (h02, c02))
        out = out[:, -1, :] #Since time steps are at the second dimension, we extract the LSTM features at the last time step
        #print("Last stride: ", out.shape)
        #Concatenating the frame count as an additional feature
#         print ('Before frame count', out.shape)
        out = torch.cat((out, frame_count.unsqueeze(dim = 1)), dim = 1).float()
#         print ('After frame count', out.shape)
        
        out = self.dropout_layer(out)
        if self.use_layernorm:
            out = self.layernorm(out)

        if self.linear_size > 1: #If there are more than one default FC layer after the LSTM layers
            out = F.tanh(self.fc0(out))
            out = self.fc1(out) #Do not use activation at the last linear layer for classification problems
        else:
            out = self.fc(out) #Do not use activation at the last linear layer for classification problems       
        return out
