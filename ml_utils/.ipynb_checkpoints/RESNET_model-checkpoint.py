from importlib import reload
import ml_utils.imports, ml_utils.padding
reload(ml_utils.imports)
reload(ml_utils.padding)
from ml_utils.imports import *
from ml_utils.padding import pad_same #For retaining the same size of the input, no matter what kernel/stride size we use
from ml_utils.positional_encoding import PositionalEncoder

'''
RESNET References: https://github.com/pytorch/vision/blob/fe973ceed96da733ec0ae61c525b2f886ccfba21/torchvision/models/resnet.py#L120-L127
'''

class BasicBlock(nn.Module):
    '''
    A Basic Block is a Resnet block with 2 Conv 1D layers (each with batchnorm and relu activation), i.e. Conv1D - Batchnorm - ReLU - Conv1D - Batch norm - Add the input x to the output (Use 1*1 conv layer to resize x if dimensions of x and output do not match) - ReLU to the added output.
    When adding the input x to the conv layers output, the channel dimensions are managed by using the 1*1 kernel conv layer on x to resize it's channels, and the shape (features * time steps) of the input are kept intact when passing through the conv layers of various kernel sizes and stride size, by using zero padding ("same" padding ensures that we pad enough in the time steps dimension, that with any kernel size and stride size, still the time steps dimension remains the same to ensure we can add it to the original input).
    Here kernel_list = [k1, k2, k3], but since we only have 2 conv layers in basic block, we use k1 as kernel for conv layer 1 and k2 for conv layer 2. Similarly, stride_list = [s1, s2, s3], but only s1 and s2 get utilized, where s1 is stride for conv layer 1 and s2 is stride for conv layer 2. 
    For pytorch default, k1 = k2 = 3 and s1 = 1 for 64 channel basic block and 2 for 128/256/512 channel basic blocks and s2 = 1 for any channel basic block.
    '''
    expansion = 1

    def __init__(self, inplanes, planes, kernel_list, stride_list, downsample=None):
        super(BasicBlock, self).__init__()
        self.kernel_size_conv1 = kernel_list[0] #3 for pytorch default
        self.stride_conv1 =  stride_list[0] #1 for 64 chans, 2 for 128/256/512 chans in pytorch default 
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.kernel_size_conv2 = kernel_list[1] #3 for pytorch default 
        self.stride_conv2 =  stride_list[1] #1 for pytorch default for all channel blocks 
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = pad_same(x, self.kernel_size_conv1, self.stride_conv1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = pad_same(out, self.kernel_size_conv2, self.stride_conv2)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

    
class Bottleneck(nn.Module):
    '''
     A Bottleneck is a Resnet block with 3 Conv 1D layers (each with batchnorm and relu activation), i.e. Conv1D - Batchnorm - ReLU - Conv1D - Batch norm - ReLU - Conv1D - Batchnorm - Add the input x to the output (Use 1*1 conv layer to resize x if dimensions of x and output do not match) - ReLU to the added output.
    When adding the input x to the conv layers output, the channel dimensions are managed by using the 1*1 kernel conv layer (given by "downsample" parameter in the __init__) on x to resize it's channels, and the shape (features * time steps) of the input are kept intact when passing through the conv layers of various kernel sizes and stride size, by using zero padding ("same" padding ensures that we pad enough in the time steps dimension, that with any kernel size and stride size, still the time steps dimension remains the same to ensure we can add it to the original input).
    Here kernel_list = [k1, k2, k3], we use k1 as kernel for conv layer 1, k2 for conv layer 2 and k3 for conv layer 3. Similarly, stride_list = [s1, s2, s3], where s1 is stride for conv layer 1, s2 is stride for conv layer 2 and s3 is stride for conv layer 3.
    For pytorch default, [k1, k2, k3] = [1, 3, 1] and s1 = 1, s2 = 1 for 64 channel bottleneck block and 2 for 128/256/512 channel basic blocks and s3 = 1 for any channel basic block.
    For strong baseline paper, https://arxiv.org/pdf/1611.06455.pdf, they use a bottleneck with [k1, k2, k3] = [8, 5, 3] and [s1, s2, s3] = [1, 1, 1]. They had one 64 channel block (with 3 conv layers as in bottleneck) and two 128 channel blocks, hence layers for the arch. in strong baseline paper is [1, 2, 0, 0], where 0 denotes no blocks with 256 channels and no blocks with 512 channels.
    Note: Because of the way we have set up our Resnet code, our strides list can have only one stride value > 1.
    '''
    expansion = 4

    def __init__(self, inplanes, planes, kernel_list, stride_list, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.kernel_size_conv1 = kernel_list[0] #1 for pytorch default 
        self.stride_conv1 = stride_list[0] #1 for pytorch default      
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=self.kernel_size_conv1, stride = self.stride_conv1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.kernel_size_conv2 = kernel_list[1] #3 for pytorch default 
        self.stride_conv2 =  stride_list[1] #1 for 64 chans, 2 for 128/256/512 chans in pytorch default 
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.kernel_size_conv3 = kernel_list[2] #1 for pytorch default 
        self.stride_conv3 = stride_list[2] #1 for pytorch default  
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=self.kernel_size_conv3, stride = self.stride_conv3, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        print ('1. In bottleneck: residual: ', residual.shape)
        out = pad_same(x, self.kernel_size_conv1, self.stride_conv1)
        print ('1. In bottleneck: After padding: ', out.shape)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        print ('AFTER CONV1. In bottleneck: residual: ', out.shape)

        out = pad_same(out, self.kernel_size_conv2, self.stride_conv2)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print ('AFTER CONV2. In bottleneck: residual: ', out.shape)

        out = pad_same(out, self.kernel_size_conv3, self.stride_conv3)
        out = self.conv3(out)
        out = self.bn3(out)
        print ('AFTER CONV3. In bottleneck: residual: ', out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)
            print ('AFTER DOWNSAMPLE. In bottleneck: residual: ', residual.shape)

        print ('END: In bottleneck: out: ', out.shape)
        print ('END: In bottleneck: residual: ', residual.shape)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
    
    in_chans: input features = 36 
    initial_conv_layer = True/False, 
    '''
    def __init__(self, in_chans, initial_conv_layer, block_name, layers, kernel_size_conv1, kernel_size_conv2, kernel_size_conv3, stride_layer64, stride_layer128, stride_layer256, stride_layer512, position_encoding = False, num_classes=3):
        super(ResNet, self).__init__()
        
        if block_name=='basic_block':
            block = BasicBlock
            kernel_list = [kernel_size_conv1, kernel_size_conv2]
        
        else: #'bottleneck'
            block = Bottleneck
            kernel_list = [kernel_size_conv1, kernel_size_conv2, kernel_size_conv3]
            
        self.in_chans = in_chans
        self.initial_conv_layer = initial_conv_layer
        if self.initial_conv_layer:
            self.inplanes = 64
        else:
            self.inplanes = self.in_chans
            
        self.position_encoding = position_encoding
        #Positional encoding to the input features 
        if self.position_encoding:
            self.pos_encoding_layer = PositionalEncoder(self.in_chans)
            
        self.conv1 = nn.Conv1d(self.in_chans, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        resnet_channel_list = [64, 128, 256, 512]
        self.layer1 = self._make_layer(block, resnet_channel_list[0], layers[0], kernel_list, stride_layer64)
        self.layer2 = self._make_layer(block, resnet_channel_list[1], layers[1], kernel_list, stride_layer128)
        self.layer3 = self._make_layer(block, resnet_channel_list[2], layers[2], kernel_list, stride_layer256)
        self.layer4 = self._make_layer(block, resnet_channel_list[3], layers[3], kernel_list, stride_layer512)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #Since if layers[i]>0, we create those many resnet blocks with corresponding channels in resnet_channel_list[i]
        created_layers = [i for i, x in enumerate(layers) if x > 0]
        #Indices for which the resnet blocks were created and thus the resnet_channel_list at the last index 
        #denotes the output channels from our Resnet network and thus input channels for the dense network
        resnet_out_channels = resnet_channel_list[created_layers[-1]]
        self.fc = nn.Linear(resnet_out_channels * block.expansion + 1, num_classes) #+1 for the extra frame count feature 

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, kernel_list, stride_list):
        downsample = None
        try:
            stride = [s for s in stride_list if s>1][0]
        except:
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        if blocks>0: #In case we do not want a resnet with this no. of channels, then layers[i] = 0
            layers.append(block(self.inplanes, planes, kernel_list, stride_list, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_list, stride_list))

        return nn.Sequential(*layers)

    def forward(self, body_coords, frame_count):
        #Positional encoding 
        if self.position_encoding:
            body_coords = self.pos_encoding_layer(body_coords)
        
        # At input, boody coords is shaped as batch_size x sequence length (time steps) x 36 features 
        #But for 1D CNN, the input should be shaped as batch_size x features/channels x time steps/sequence length
        x = body_coords.permute(0,2,1)
        print ('IN MAIN RESNET:', x.shape)
        if self.initial_conv_layer:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            print ('IN MAIN RESNET: AFTER THE INITIAL CONV LAYER', x.shape)
        
        x = self.layer1(x)
        print ('IN MAIN RESNET: AFTER THE FIRST RESNET LAYER', x.shape)
        x = self.layer2(x)
        print ('IN MAIN RESNET: AFTER THE SECOND RESNET LAYER', x.shape)
        x = self.layer3(x)
        print ('IN MAIN RESNET: AFTER THE THIRD RESNET LAYER', x.shape)
        x = self.layer4(x)
        print ('IN MAIN RESNET: AFTER THE FOURTH RESNET LAYER', x.shape)

        x = self.avgpool(x)
        print ('IN MAIN RESNET: AFTER THE AVGPOOL LAYER', x.shape)
        x = x.view(x.size(0), -1)
        #Adding the extra frame count feature before using the fully connected layers 
        x = torch.cat((x, frame_count.unsqueeze(dim = 1)), dim = 1).float()  
        print ('IN MAIN RESNET: AFTER THE FLATTEN AND CONCAT', x.shape)
        x = self.fc(x)
        print ('IN MAIN RESNET: AFTER THE FULLY CONNECTED', x.shape)

        return x