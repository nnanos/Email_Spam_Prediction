import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pytorch_model_summary import summary



class Model(nn.Module):
    def __init__(self, input_size , hidden_size , output_size , model_type ):

        super(Model, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.model_type = model_type

        if self.model_type=="RNN":
            self.rnn = nn.RNN(input_size , hidden_size, num_layers=1, bidirectional=False)            
        else:
            self.rnn = nn.LSTM(input_size , hidden_size, num_layers=1, bidirectional=False)

        self.label = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax()
	
    def forward(self,input_batch):

        if self.model_type=="RNN":
            output, h_n = self.rnn(input_batch)
        else:
            output, (h_n , _) = self.rnn(input_batch)

        logits = self.label(h_n) # logits.size() = (batch_size, output_size)        


        #out = self.softmax(logits)


        return logits


class LSTM(nn.Module):
    def __init__(self, input_size , hidden_size , output_size , batch_size ):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size , hidden_size, num_layers=1, bidirectional=False)
        self.linear = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax()
	
    def forward(self,input_batch,h_0,c_0):
        
        #input = input.permute(1, 0, 2)

        #output, h_n = self.rnn(input_batch, h_0)
        output, (h_n, cn) = self.rnn(input_batch, (h_0, c_0))

        logits = self.linear(h_n) # logits.size() = (batch_size, output_size)

        #out = self.softmax(logits)


        return logits



def create_conf_matrix(Y_hat,Y_target):

    total_num_of_pred_spams = Y_hat.sum()
    total_num_of_pred_nonspams = len(Y_hat) - total_num_of_pred_spams

    Y = np.concatenate((np.array([Y_hat]),Y_target))    
   
    #counting the number of 2 (thats the True positive)
    tmp = Y.sum(0)

    Tp = len(tmp[tmp==2])
    Fp = total_num_of_pred_spams - Tp

    Tn = len(tmp[tmp==0])
    Fn = total_num_of_pred_nonspams - Tn

    return np.array([[Tp,Fn],[Fp,Tn]])



#print( summary(Net()) )

#unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)