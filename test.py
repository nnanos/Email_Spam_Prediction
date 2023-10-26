import pandas as pd
import numpy as np
import gensim
import gensim.downloader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
#import utils
from model import RNN
import tqdm
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import timeit
from load_data_functions import * 
from sklearn.model_selection import train_test_split




def save_model(model,epoch,optimizer,loss,PATH):
    #Saving & Loading a General Checkpoint for Inference and/or Resuming Training
    torch.save( {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    
    } , PATH )

    #in order to load and resume training do the following:
    
    '''
    model = TheModelClass(*args, **kwargs)
    optimizer = TheOptimizerClass(*args, **kwargs)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    # - or -
    model.train()
    '''




def train_and_save_w2v_model(X):
    #A-> array of strings (emails)
    #train a word2vec model with the email dataset

    #s_time = time.time()
    print("Model Training started")
    model = Word2Vec(sentences=X, vector_size=300, window=6, min_count=1, workers=5)
    model.save("word2vec.model")
    #cal_elapsed_time(s_time)

   
    
def train( model , train_sampler, critireon , optimizer ):

    #Define the initial state of the RNN
    h_0 = np.zeros((1,batch_size,hidden_size))
    h_0 = torch.zeros((1,batch_size,hidden_size)).to(device)    
    
    model.train()
    pbar = tqdm.tqdm(train_sampler)
    train_losses = []
    #AT EACH ITERATION WE FEED IN A BATCH OF SEQUENCES TO THE RNN AND THEN WE BACKPROPAGATE
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        Y_hat = model(x,h_0)

        #CROSSENTROPY EXPECTS THE INPUTS TO BE 
        Y_hat = Y_hat.squeeze()#Input: (BATCH,nb_CLASSES)
        y = y.squeeze() #TARGET:  (BATCH)

        loss = critireon(Y_hat, y)
        train_losses.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()


    return  train_losses


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

def validate(model, val_sampler,critireon):
    sigmoid = nn.Sigmoid()

    #Define the initial state of the RNN
    #
    h_0 = np.zeros((1,1,hidden_size))
    h_0 = torch.zeros((1,1,hidden_size)).to(device)    
    
    Y_target = np.array([val_sampler[1]])
    valid_losses = []
    Y_hat = []
    model.eval()
    with torch.no_grad():
        for x in val_sampler[0]:
            x = x.to(device)

            Y_hat.append( sigmoid(model(x,h_0)).cpu().detach().numpy() )

        Y_hat = np.array(Y_hat).reshape(-1)
        Y_hat[Y_hat>0.5]=1
        Y_hat[Y_hat<0.5]=0

    M = create_conf_matrix(Y_hat,Y_target)

    Precision = M[0,0] / M[:,0].sum()
    Recall = M[0,0] / M[0,:].sum()        
            
    return Recall,Precision

def validate1(model, val_sampler,critireon):
    sigmoid = nn.Sigmoid()

    #Define the initial state of the RNN
    h_0 = np.zeros((1,batch_size,hidden_size))
    h_0 = torch.zeros((1,batch_size,hidden_size)).to(device)    
    
    valid_losses = []
    model.eval()
    with torch.no_grad():
        for x, y in val_sampler:
            x, y = x.to(device), y.to(device)

            Y_hat = model(x,h_0)
            Y_hat_true = sigmoid(Y_hat) 

            Y_hat = Y_hat.squeeze()#Input: (BATCH,nb_CLASSES)
            y = y.squeeze() #TARGET:  (BATCH)

            loss = critireon(Y_hat, y)
            valid_losses.append(loss.cpu().detach().numpy())

            
            
    return valid_losses

# Download the "glove-twitter-25" embeddings
#glove_vectors = gensim.downloader.load('glove-twitter-25')
# Load vectors directly from the file
#model = KeyedVectors.load_word2vec_format('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΕΞΟΡΥΞΗ_ΔΕΔΟΜΕΝΩΝ/DM_project_2021/GoogleNews-vectors-negative300.bin', binary=True)

if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #EACH EMAIL IS SPLITTED TO IT'S CORRESPONDING WORDS
    X,Y = load_data()

    x = int(input('Select Menu:\n0)Train a custom word2vec model on the dataset\n1)Load the google pretrained word2vec model\n:'))
    if not(x):
        #custom
        #train_and_save_w2v_model(X)
        w2v_model = Word2Vec.load("word2vec.model")
        model_type = 'custom'
    else:
        #google
        start = timeit.timeit()
        w2v_model = KeyedVectors.load_word2vec_format('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΕΞΟΡΥΞΗ_ΔΕΔΟΜΕΝΩΝ/DM_project_2021/GoogleNews-vectors-negative300.bin', binary=True)
        end = timeit.timeit()
        print(end - start)
        '''
        start = timeit.timeit()
        w2v_model = KeyedVectors.load_word2vec_format('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΕΞΟΡΥΞΗ_ΔΕΔΟΜΕΝΩΝ/DM_project_2021/GoogleNews-vectors-negative300.bin', binary=True)
        end = timeit.timeit()
        print(end - start)
        '''

        model_type = 'pretrained'

    # Download the "glove-twitter-25" embeddings
    #glove_vectors = gensim.downloader.load('glove-twitter-25')

    
    
    #using packed sequences with variable length 
    #batch_size = 50
    hidden_size = 128 #d
    input_size = 300
    output_size = 1
    epochs = 1

    #GENERATING TRAINING AND TESTING DATA---------------------------------------------------------------
    embeded_emails , inds_removed = get_embeded_emails(w2v_model,model_type,X)

    if not(inds_removed.size == 0):
        Y = np.delete( Y , inds_removed )

     
    X_train, X_test, Y_train, Y_test = train_test_split(embeded_emails, Y, test_size=0.20, random_state=42)

    #find the max number (in a range) that divides perfect the number of examples-------
    for d in range(2,150+1):
        if not( ( len(X_train) % d ) ) and ( d>30 and d<=200) :
            batch_size = d
        
    try:
        print(batch_size)
    except:
        batch_size = 1
        print("Didnt find any apropriate batch_size...using batch_size=1 ")
    #---------------------------------------------------------------------------

    #Y_test , Y_train = Y[int(1500*0.80):] , Y[:int(1500*0.80)]
    #splitting the dataset to training and testing emails (80%->train,20%->testing)
    #X_train , X_test =  embeded_emails[:int(1500*0.80)] , embeded_emails[int(1500*0.80):]
    train_sampler = get_training_batches(X_train,Y_train,batch_size)
    valid_sampler = ( list( map(lambda x: torch.FloatTensor(x).unsqueeze(1) , X_test ) ) , Y_test )
    #valid_sampler = get_training_batches(X_test,Y_test,len(Y_test))
    #valid_sampler = get_training_batches(X_test,Y_test,batch_size)


    #DEFINE MODEL,LOSS,OPTIMIZER---------------------------------------------------------------
    rnn_model = RNN(input_size,hidden_size,output_size,batch_size).to(device)
    #critireon = nn.CrossEntropyLoss()
    critireon = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(rnn_model.parameters(), lr=0.001)
    #train( rnn_model , train_sampler , critireon , optimizer)
    #valid_loss = validate(rnn_model,valid_sampler,critireon)
    #--------------------------------------------------------------------------------------------

    #TRAINING LOOP-------------------------------------------------------------------------
    t = tqdm.trange(1, epochs + 1)

    learning_curve = []
    for epoch in t:
        train_loss_tmp = train( rnn_model , train_sampler , critireon , optimizer) 
        learning_curve.append(train_loss_tmp)

    #for each epoch we average the loss of all the mini-batches to obtain the learning curve (Loss vs epoch)
    learning_curve = np.array( list( map(lambda x: np.array(x).mean() , learning_curve) ) )
    #----------------------------------------------------------------------------------------

    #TESTING---------------------------------------------------------------------------------------
    valid_loss = validate(rnn_model,valid_sampler,critireon)



    #PRINT MODEL SUMMARY---------------------------------
    PATH = '/home/nnanos/Desktop/PYTHON/eksoruksh_dedomenwn/drive-download-20210509T145040Z-001/model_checkpoint/model_check'
    save_model(rnn_model, epoch, optimizer, learning_curve, PATH)
    #print(summary(RNN(input_size,hidden_size,output_size,batch_size),torch.zeros((input_size, batch_size, input_size)), show_input=False, show_hierarchical=True))

a = 0


#LSTM TEST--------------------------------------------------------------------
'''
input = train_sampler[0][0]
label = train_sampler[0][1]
input = tmp[0][0]
input = np.expand_dims(input, axis=1) #adding a siglenton dim for the batch size

batch_size = 1
hidden_size = 128
#input of shape (seq_len, batch, input_size)
input_size = input.shape[2]
#h_0 of shape (num_layers * num_directions, batch, hidden_size)
h_0 = np.zeros((1,batch_size,hidden_size))
h_0 = torch.FloatTensor(h_0)


input = torch.FloatTensor(input)
rnn = nn.RNN(input_size, hidden_size)
out , hn = rnn(input,h_0)
'''


'''
#h_0 of shape (num_layers * num_directions, batch, hidden_size)
h_0 = np.zeros((1,batch_size,hidden_size))
h_0 = torch.FloatTensor(h_0)
rnn = nn.RNN(input_size, hidden_size)
out , hn = rnn(input,h_0)
unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)
'''