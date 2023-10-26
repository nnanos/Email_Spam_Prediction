import torch.nn as nn
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import timeit
from sklearn.model_selection import train_test_split
import json
import utils
import argparse
from pathlib import Path
import copy
import random
import time
import os
from model import *
import pickle
from load_data_functions import generate_and_save_trainable_dataset


def get_learning_curve(train_losses,valid_losses,best_epoch):

    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.xlabel("epochs")
    #Avg LOSS (over all the minibatches)
    plt.ylabel("Avg LOSS")
    plt.axvline(x=best_epoch , linestyle='--', color='b',label='Best_epoch')
    plt.legend(["Training_Loss", "Validation_loss","Best_epoch"], loc ="upper left")
    plt.show()

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

def validate(model,device, val_sampler):
    sigmoid = nn.Sigmoid()
    
    Y_target = np.array([val_sampler[1]])
    valid_losses = []
    Y_hat = []
    model.eval()
    with torch.no_grad():
        for x in val_sampler[0]:
            x = x.to(device)

            Y_hat.append( sigmoid( model(x) ).cpu().detach().numpy() )

        Y_hat = np.array(Y_hat).reshape(-1)
        Y_hat[Y_hat>0.5]=1
        Y_hat[Y_hat<0.5]=0

    M = create_conf_matrix(Y_hat,Y_target)

    Precision = M[0,0] / M[:,0].sum()
    Recall = M[0,0] / M[0,:].sum()
    F1 = 2*(Precision*Recall)/(Precision+Recall)        
            
    return Recall,Precision,F1

if __name__ == "__main__":


    #EVALUATE---------------------------------------
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model', type=str, help='Path to checkpoint folder', default="/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/PROJECTS_TO_TEST/CEID_PROJECTS/MACHINE_LEARNING_PROJECTS/Spam_or_not_spam_Neural_Net-main/output_fold")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--Model_type', type=str, default="RNN",
                        help='0)RNN\n1)LSTM (default: RNN)')
    parser.add_argument('--root', default="shuffled_dataset.csv" , type=str, 
                    help='root path of raw emails, if none then it is assumed that in the pwd there exists a variable (dataset_dict) that contains all the appropriate vars to load ')


    args, _ = parser.parse_known_args()

    #LOADING THE MODEL AND THE TRAIN VALIDATION LOSS-----------------------

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")


    #Open the folder that contains the :
    # json log cht (contains all the hypers used , the train-valid loss and time per iter)
    # .chkpnt (contains all the state dicts for the model,optimizer,scheduler of the last epoch) 
    # and the .pth (contains the model state dict of the best epoch)
    model_path = Path(args.model).expanduser()
    with open( Path(model_path, args.Model_type + '.json'), 'r' ) as stream:
        check_json = json.load(stream)

    target_model_path = Path(model_path, args.Model_type + ".pth")
    checkpoint = torch.load(target_model_path, map_location=device)

    rnn_model = Model(
        check_json["args"]["input_size"],
        check_json["args"]["hidden_size"],
        check_json["args"]["output_size"],
        check_json["args"]["Model_type"]
    ).to(device)

    rnn_model.load_state_dict(checkpoint)



    # #MINI model summary----
    #  print(rnn_model)
    #  print( "total number of trainable parameters : {}" .format(sum([param.nelement() for param in rnn_model.parameters()])))
    # #--------------------------------

    train_losses = check_json['train_loss_history']
    valid_losses = check_json['valid_loss_history']
    train_times = check_json['train_time_history']
    best_epoch = check_json['best_epoch']
    # es.best = results['best_loss']
    # es.num_bad_epochs = results['num_bad_epochs']



    #EVALUATION ON THE TESTING SET AND PLOTTING THE LEARNING CURVES(the last is refering to the training phase of the NN)--------------------------------

    get_learning_curve(train_losses,valid_losses,best_epoch)

    '''
    # Lodaing the dataset dict
    with open('dataset_and_batch_size_dict.pkl' , 'rb') as f:  # Python 3: open(..., 'rb')
        dataset_dict = pickle.load(f)
    '''

    #Generate the trainable dataset (i.e. word embeddings , batch size etc)    
    dataset_dict = generate_and_save_trainable_dataset(args.root , word2vec_flag=1)
    
    test_sampler = ( list( map(lambda x: torch.FloatTensor(x).unsqueeze(1) , dataset_dict["test"][0] ) ) , dataset_dict["test"][1] )

    Recall,Precision,F1 = validate(rnn_model,device,test_sampler)
    # save evaluation metrics
    metrics = {
        'Recall' : Recall,
        'Precision' : Precision,
        'F1_score' : F1
    }
    print(metrics)

    '''
    with open(Path_to_json_log, 'a') as outfile:
        outfile.write(json.dumps(metrics, indent=4, sort_keys=True))
    '''
    #----------------------------------------------------------------------