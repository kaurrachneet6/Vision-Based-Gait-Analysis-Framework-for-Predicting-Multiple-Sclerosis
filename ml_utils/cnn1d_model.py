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
#         print ('x', x)
#         print ('\nx shape', x.shape)
#         print ('\nx type', x.type())
#         print ('\nIn CNN model, frame count: ', frame_count)
#         print ('\nframe count shape', frame_count.shape)
#         print ('\n frame count type', frame_count.type())
        x = torch.cat((x, frame_count.unsqueeze(dim = 1)), dim = 1).float()
#         print ('new x', x)
#         print ('new x shape', x.shape)
#         print ('new x type', x.type())
        x = F.relu(self.fc(x))
#         print (x, x.shape)
#         probs = F.softmax(x, dim=1)         
        return x