import pandas as pd
import numpy as np
import gensim
import gensim.downloader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
#import utils
from model import *
import tqdm
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import timeit
from load_data_functions import * 
from evaluation import *
from sklearn.model_selection import train_test_split
import json
import utils
import argparse
from pathlib import Path
import copy
import random
import time
import os
import pickle


tqdm.monitor_interval = 0


def train(args, model, device, train_sampler, critireon , optimizer):
    losses = utils.AverageMeter()
    model.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        Y_hat = model(x)
        Y_hat = Y_hat.squeeze()#Input: (BATCH,nb_CLASSES)
        y = y.squeeze() #TARGET:  (BATCH)
        loss = critireon(Y_hat, y)

        loss.backward()
        optimizer.step()
        #losses.update(loss.item(), y.size(1))
        losses.update(loss.item())
    return losses.avg


def valid(args, model, device, val_sampler , critireon):
    losses = utils.AverageMeter()

    Y_target = torch.FloatTensor(np.array([val_sampler[1]])).to(device)
    model.eval()
    with torch.no_grad():
        i = 0
        for x in val_sampler[0]:
            

            x, y = x.to(device) , Y_target[:,i]

            Y_hat = model(x)
            
            Y_hat = Y_hat.squeeze(2)
            y = y.unsqueeze(0)

            loss = critireon(Y_hat, y)

            losses.update(loss.item())
            i+=1
        return losses.avg


def main():
    
    parser = argparse.ArgumentParser(description='Trainer')

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="small_dataset",
                        choices=[
                            'small_dataset', 'big_dataset'
                        ],
                        help='Name of the dataset.')
    parser.add_argument('--root', default="shuffled_dataset.csv" , type=str, 
                    help='root path of raw emails, if none then it is assumed that in the pwd there exists a variable (dataset_dict) that contains all the appropriate vars to load ')
    parser.add_argument('--output', type=str, default="output_fold",
                        help='provide output path base folder name')
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=10)
    #parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=10,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')


    # Model Parameters
    # parser.add_argument('--word2vec_model_path', type=str,
    #                     help='1)Load a pretrained Custom_word2vec model on the dataset\n0)Load a pretrained word2vec model on google dataset')
                        
    parser.add_argument('--word2vec_flag', type=bool, default=1,
                        help='1)Load a pretrained Custom_word2vec model on the dataset\n0)Load a pretrained word2vec model on google dataset')
    parser.add_argument('--unidirectional', action='store_true', default=True,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--Model_type', type=str, default="RNN",
                        help='0)RNN\n1)LSTM (default: RNN)')
    parser.add_argument('--input_size', type=int, default=300,
                        help='input dimensionality (default: 300)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='state space dimensionality (default: 128)')
    parser.add_argument('--output_size', type=int, default=1,
                        help='output dimensionality (default: 1)')


    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    #print("Using Torchaudio: ", utils._torchaudio_available())

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    #GENERATING TRAINING VALIDATION and Testing DATA---------------------------------------------------------------
    #EACH EMAIL IS SPLITTED TO IT'S CORRESPONDING WORDS

    #Generate the trainable dataset (i.e. word embeddings , batch size etc)    
    dataset_dict = generate_and_save_trainable_dataset(args.root,args.word2vec_flag)


    train_sampler = get_training_batches(dataset_dict["train"][0],dataset_dict["train"][1],dataset_dict["batch_size"])
    valid_sampler = ( list( map(lambda x: torch.FloatTensor(x).unsqueeze(1) , dataset_dict["valid"][0] ) ) , dataset_dict["valid"][1] )
    test_sampler = ( list( map(lambda x: torch.FloatTensor(x).unsqueeze(1) , dataset_dict["test"][0] ) ) , dataset_dict["test"][1] )
    #---------------------------------------------------------------------------------------------------------



    rnn_model = Model(
        args.input_size,
        args.hidden_size,
        args.output_size,
        args.Model_type
    ).to(device)

    critireon = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        rnn_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )


    es = utils.EarlyStopping(patience=args.patience)

    #DECIDING IF WE ARE GONNA CONTINUE TRAINING OR START FROM THE BEGGINING--------------------
    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open( Path(model_path, args.Model_type + '.json'), 'r' ) as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.Model_type + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        rnn_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    #-----------------------------------------------------------------------------------
    

    #TRAINING LOOP----------------------------------------------------------------
    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(args, rnn_model, device, train_sampler, critireon , optimizer)
        valid_loss = valid(args, rnn_model, device, valid_sampler,critireon)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        stop = es.step(valid_loss)


        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': rnn_model.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
            model_type = args.Model_type
        )

    
        # save params
        vars(args)["batch_size"]=dataset_dict["batch_size"]
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            #'commit': commit
        }

        Path_to_json_log = Path(target_path,  args.Model_type + '.json')

        with open(Path_to_json_log, 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break
    #--------------------------------------------------------------------------

    '''
    #EVALUATION ON THE TESTING SET (when the training is done)----------------------------------------
    Recall,Precision,F1 = validate(rnn_model,device,test_sampler)
    # save evaluation metrics
    metrics = {
        'Recall' : Recall,
        'Precision' : Precision,
        'F1_score' : F1
    }
    print(metrics)
    with open(Path_to_json_log, 'a') as outfile:
        outfile.write(json.dumps(metrics, indent=4, sort_keys=True))
    #----------------------------------------------------------------------
    '''

    a = 0

if __name__ == "__main__":
    main()
    #print(summary(RNN(input_size,hidden_size,output_size,batch_size),torch.zeros((input_size, batch_size, input_size)), show_input=False, show_hierarchical=True))
    a = 0