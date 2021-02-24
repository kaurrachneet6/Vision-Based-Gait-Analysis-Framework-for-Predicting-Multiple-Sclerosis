from importlib import reload
import ml_utils.imports
reload(ml_utils.imports)
from ml_utils.imports import *

'''
Positional Encoding Reference: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
'''

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        #Refer https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
        #for the formula of sin-cos positional encoding 
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        #Adding the positional encoding to the original features 
        x = x + Variable(self.pe[:, :seq_len], \
        requires_grad=False)
        return x