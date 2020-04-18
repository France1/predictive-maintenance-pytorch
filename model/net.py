import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_len, input_dim=9, output_dim=3):
        """Pytorch 1D CNN model for time series classification. Model architecture from:
        Ye Yuan, et al., A general end-to-end diagnosis framework for manufacturing systems, National Science Review, Volume 7, Issue 2, February 2020, Pages 418–429, https://doi.org/10.1093/nsr/nwz190       
        
        Arguments:
            in_len : int
                length of time sequence
            input_dim : int
                number of channels (sensor time sequences)
            output_dim : int
                number of classification labels       
        """
        super(CNN, self).__init__()
        self.in_len = in_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 10, 5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(self.fc_len(), 500), # calculate input dimension adaptively
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 50),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(50, self.output_dim)

    def forward(self, x):
        """Forward pass of model network       
        
        Inputs:
            x: pytorch tensor (batch, channels, sequence)
                batch of input data
            
        Outputs:
            x: pytorch tensor (batch, labels)
                batch of labels
        """
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        x = self.classifier(x)
        return x
    
    def out_len_conv(self, in_len, conv_layer):
        """Calculate output length of 1D conv layer       
        
        Inputs:
            in_len: int
                input dimension of conv layer
            conv_layer: object
                pytorch conv1d 
            
        Outputs:
            out_len: int
                output dimension of conv layer
        """
        out_len = (in_len-conv_layer.kernel_size[0]+2*conv_layer.padding[0])/conv_layer.stride[0]+1
        return out_len

    def fc_len(self):
        """Calculate output length of conv_block for linear layer connection       
        """
        out = self.out_len_conv(self.in_len, self.conv_block[0])
        out = int(out/2)
        out = self.out_len_conv(out, self.conv_block[4])    
        out = int(out/2)
        out = out*self.conv_block[4].out_channels
        return out
    

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=2, output_dim=1):
        """Pytorch vanilla LSTM model for time series classification       
        
        Arguments:
            input_dim : int
                number of channels (sensor time sequences) 
            hidden_dim : int
                hidden layer size
            num_layers : int
                number of layers in LSTM block
            output_dim : int
                number of classification labels        
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        
        self.fc_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        ) 
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        """Forward pass of model network       
        
        Inputs:
            input: pytorch tensor (batch, channels, sequence)
                batch of input data
            
        Outputs:
            out: pytorch tensor (batch, labels)
                batch of labels
        """
        out, hidden = self.lstm(input.permute(2,0,1)) # (batch, channels, sequence) -> [sequence, batch, channels]
        out = self.fc_block(out[-1])
        out = self.classifier(out)
        return out


class LSTMattn(nn.Module):
    """Pytorch LSTM model with attention for time series classification. 
    Attention model from:
    M.Luong et al, Effective Approaches to Attention-based Neural Machine Translation, 2015, arXiv:1508.04025
    Implementation from:
    https://github.com/prakashpandey9/Text-Classification-Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation 
    
        Arguments:
            input_dim : int
                number of channels (sensor time sequences) 
            hidden_dim : int
                hidden layer size
            num_layers : int
                number of layers in LSTM block
            output_dim : int
                number of classification labels        
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1):
        super(LSTMattn, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.8)
        self.fc_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        ) 
        self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)   
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
    
    def attention(self, lstm_output, hidden):
        """Luong attention model for sequence classification       
        
        Inputs:
            lstm_output: pytorch tensor (sequence, batch, hidden)
                output of LSTM 
            hidden: pytorch tensor (batch, hidden)
                hidden state of LSTM
            
        Outputs:
            output: pytorch tensor (batch, hidden)
                hidden state with applied attention
        """
        hidden = hidden.squeeze(0)
        lstm_output = lstm_output.permute(1,0,2)

        scores = torch.bmm(lstm_output, hidden.unsqueeze(2))
        attn_weights = F.softmax(scores, 1) # eq.7 
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weights).squeeze(2)
        
        concat_input = torch.cat((hidden, context), 1)
        output = torch.tanh(self.concat(concat_input)) # eq. 5
        
        return output

    def forward(self, input):
        """Forward pass of model network       
        
        Inputs:
            input: pytorch tensor (batch, channels, sequence)
                batch of input data
            
        Outputs:
            out: pytorch tensor (batch, labels)
                batch of labels
        """
        input = input.permute(2,0,1)
        lstm_out, (h,c) = self.lstm(input)
        out = self.attention(lstm_out, h[-1])
        out = self.classifier(out)
        
        return out
    

class MultiClassifier(nn.Module):
    """Pytorch multi task classifier block for pump maintenance dataset. 
    See:
    https://discuss.pytorch.org/t/how-to-do-multi-task-training/14879/7
    https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681/2
    """
    def __init__(self, input_dim):
        super(MultiClassifier, self).__init__()
        self.fc_0   = nn.Linear(input_dim, 3)
        self.fc_1   = nn.Linear(input_dim, 4)
        self.fc_2   = nn.Linear(input_dim, 3)
        self.fc_3   = nn.Linear(input_dim, 4)

    def forward(self, x):
        x_0 = self.fc_0(x)
        x_1 = self.fc_1(x)
        x_2 = self.fc_2(x)
        x_3 = self.fc_3(x)
        
        return x_0, x_1, x_2, x_3