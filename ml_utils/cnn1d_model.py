from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *


class CNN1D(nn.Module):
    def __init__(self, in_chans, out_chans, dropout):
        super(CNN1D, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dropout = dropout
        
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=50, kernel_size=5, stride=2)
        self.conv1.weight = nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = nn.Conv1d(in_channels=50, out_channels=10, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(161, 3)
    
    def forward(self, body_coords, frame_count):
#         print (body_coords)
        x = F.relu(self.conv1(body_coords))
#         print (x.shape)
        x = F.relu(self.conv2(x))
#         print (x.shape)
        x = x.view(-1, x.shape[1]*x.shape[2])
#         print (x.shape)
        print ('x', x)
        print ('\nx shape', x.shape)
        print ('\nx type', x.type())
        print ('\nframe count', frame_count)
        print ('\nframe count shape', frame_count.shape)
        print ('\n frame count type', frame_count.type())
        x = torch.cat((x, frame_count.unsqueeze(dim = 1)), dim = 1).float()
        print ('new x', x)
        print ('new x shape', x.shape)
        print ('new x type', x.type())
        x = F.relu(self.fc(x))
#         print (x, x.shape)
#         probs = F.softmax(x, dim=1)


# out = out[:,-1,:]
#         #print("Last stride: ", out.shape)
#         #print("fixed shape: ", out.shape)
#         out = self.dropout(out)
#         #print("dropout shape: ", out.shape)
#         #print(out)
#         #print("after dropout: ", out.shape)
#         #print("fc expected size: ", hidden_size2*bi_mult2)
#         #out = self.fc(out[:,:-1])
#         #out = F.tanh(self.fc0(out))
#         out = F.tanh(self.fc(out))
        
        
        return x
    

#dimentions among other things are not correct
class LSTM(nn.Module):
    '''
    Pytorch LSTM model class
    Functions:
        init: initializes model based on given parameters
        forward: forward step through model
    '''
    def __init__(self, input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional1, bidirectional2, pre_out, single_lstm, linear_size):
        super(LSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers1 = num_layers1
        self.input_size = input_size
        self.hidden_size2 = hidden_size2
        self.num_layers2 = num_layers2
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional1 = bidirectional1
        self.bidirectional2 = bidirectional2
        self.pre_out = pre_out
        self.single_lstm = single_lstm
        self.linear_size = linear_size
        #self.bi_mult1 = 1
        #self.bi_mult2 = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Try splitting lstm into multiple with different hidden sizes 
        
        if (self.bidirectional1):
            bi_mult1 = 2
        else:
            bi_mult1 = 1      

        if (self.bidirectional2):
            bi_mult2 = 2
        else:
            bi_mult2 = 1
        
        if (self.single_lstm):
            hidden_size = hidden_size1
        else:
            hidden_size = hidden_size2

        self.prefc = nn.Linear(input_size, pre_out)
        #Does num_layers actually make multiple lstm layers?
        #self.lstm = nn.LSTM(pre_out, hidden_size1, num_layers1, batch_first=True, bidirectional = bidirectional1) #uncomment for first linear layer

        ##Use these two for two seperate lstm layers
        self.lstm = nn.LSTM(input_size, hidden_size1, num_layers1, batch_first=True, bidirectional = bidirectional1) #uncomment for first lstm layer

        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional = bidirectional) 
        self.lstm2 = nn.LSTM(hidden_size1*bi_mult1, hidden_size2, num_layers2, batch_first=True, bidirectional = bidirectional2)

        #Batch_first means shape is (batch, seq, feature)
        self.dropout = nn.Dropout(p=dropout)
        #self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        #self.fc = nn.Linear(hidden_size*2, 1) replaced with below

        self.fc0 = nn.Linear(hidden_size*bi_mult2, linear_size)

        #self.fc = nn.Linear(linear_size, 2)

        self.fc = nn.Linear(hidden_size*bi_mult2, 2)



        #print("fc constructor: ",hidden_size2*bi_mult2)
        #self.fc2 = nn.Linear(sequence_length, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(x.shape)
        #print(x)
        #print(x.type())
        #print(x[0].type())
        #print(x[0][0].type())
        
        set_random_seed(seed)
        
        # Set initial hidden and cell states
        #Use xavier initializtion instead of zeros
        #Or something else fancier: kaiming he
        if (self.bidirectional1):
            bi_mult1 = 2
        else:
            bi_mult1 = 1        

        if (self.bidirectional2):
            bi_mult2 = 2
        else:
            bi_mult2 = 1
        
        h = torch.empty(self.num_layers1*bi_mult1, x.size(0), self.hidden_size1)
        c = torch.empty(self.num_layers1*bi_mult1, x.size(0), self.hidden_size1)
        
        h0 = torch.nn.init.xavier_normal_(h, gain=1.0).to(device)
        c0 = torch.nn.init.xavier_normal_(c, gain=1.0).to(device)
        
        #h0 = torch.nn.init.kaiming_uniform_(h, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c0 = torch.nn.init.kaiming_uniform_(c, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        h2 = torch.empty(self.num_layers2*bi_mult2, x.size(0), self.hidden_size2)
        c2 = torch.empty(self.num_layers2*bi_mult2, x.size(0), self.hidden_size2)
        
        h02 = torch.nn.init.xavier_normal_(h2, gain=1.0).to(device)
        c02 = torch.nn.init.xavier_normal_(c2, gain=1.0).to(device)
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)


        #linear layer takes features in and then output as different dimention, change LSTM size to match
        #print("x shape: ", x.shape)

        #out = F.relu(self.prefc(x)) #Uncomment for first linear layer
        #out = (self.prefc(x))
        #print("prefc shape: ", out.shape)

        # Forward propagate LSTM
        #, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size) #uncomment for first linear layer
        out, _ = self.lstm(x, (h0, c0))    #uncomment for first lstm layer
        #print("lstm1 out: ", out.shape)
        if (not self.single_lstm):
            out, _ = self.lstm2(out, (h02, c02))
        #print("lstm2 out: ", out.shape)
        #out = out.reshape(out.shape[0], -1)
        #print(torch.isnan(out).any())
        #print("lstm out shape: ", out.shape)
        out = out[:,-1,:]
        #print("Last stride: ", out.shape)
        #print("fixed shape: ", out.shape)
        out = self.dropout(out)
        #print("dropout shape: ", out.shape)
        #print(out)
        #print("after dropout: ", out.shape)
        #print("fc expected size: ", hidden_size2*bi_mult2)
        #out = self.fc(out[:,:-1])
        #out = F.tanh(self.fc0(out))
        out = F.tanh(self.fc(out))
        #print("fully connected 1 shape: ", out.shape)
        #print(out)
        
        #Look at flattening instead of squeeze to reduce 2 fc layers to one
        #out = torch.squeeze(out)
        #out = F.relu(self.fc2(out))
        #print("fully connected 2 shpae: ", out.shape)
        
        #out = self.softmax(out)
        #print("softmax out shape: ", out.shape)
        #print(out)
        
        

        #check: is lstm output 10x30x2? if so it is all timesteps output, we only want the last one out[:,:,-1]? shape should be 30x2 for last one

        
        return out