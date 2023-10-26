import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#THE SEQUENCE OF EXECUTION OF THE FUNCTIONS BELOW IS FROM UP TO DOWN!!!!!!

def load_data(Path):
    #A = pd.read_csv('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΕΞΟΡΥΞΗ_ΔΕΔΟΜΕΝΩΝ/DM_project_2021/spam_or_not_spam/spam_or_not_spam.csv').to_numpy()
    data = pd.read_csv(Path)
    #replacing the empty email with the empty string
    #data.loc[1466,'email'] = ''
    data = data.fillna('')
    #drop a random recodr in order to have even number of records (for the division of the batch size)
    #data = data.drop([np.floor(np.random.uniform(1,1500))])
    Y =  pd.to_numeric(data['label'], errors='coerce').to_numpy()
    #Y = np.array( [float(Y[i]) for i in Y ] )
    X = data.drop('label',axis=1).to_numpy().reshape(-1)
    result = map(lambda x: x.split(" ")[:-1] , X)
    #X contains a list whose elements are np arrays that represents an emails words
    X = np.array(list(result))

    return X,Y



def get_embeded_emails(model_obj,model_type,X):
    #CREATING the word embedings fo each email-------------------------------------------
    embeded_emails = []

    if model_type=='pretrained':

        def get_word2vec_pre_trained(model_obj,word):
            try:
                return model_obj[word]
            except:
                #if there is no matching word on the vocabualry of the pretrained model
                return np.zeros(300)


        for email in X:
            #tmp_embeded_email = np.array( list( map( lambda x: get_word2vec_pre_trained(model_obj,x) , email ) ) )
            #remove all nones
            tmp_embeded_email = []
            for word in email:
                tmp = get_word2vec_pre_trained(model_obj,word)
                if tmp.sum():
                    tmp_embeded_email.append(tmp)

            embeded_emails.append( np.array(tmp_embeded_email) )


    else:

        for email in X:
            tmp_embeded_email = np.array( list( map( lambda x: model_obj.wv[x] , email ) ) )
            nb_words_per_email = tmp_embeded_email.shape[0]

            embeded_emails.append( tmp_embeded_email )
            #inds = np.array([])
    #----------------------------------------------------------------------------------

    #REMOVE EMPTY email embedings-------------------------------------[
    #IF THE SUM OF ALL ELEMENTS OF AN EMBEDED EMAIL IS 0 THEN ITS PROBABLY AN EMPTY ARRAY ( np.array([[]]) )
    a = np.array( list( map( lambda x: x.sum() , embeded_emails ) ) )
    inds = np.where(a == 0)[0]
    embeded_emails = list( np.delete( np.array(embeded_emails) , inds ) )

    return embeded_emails , inds



#def get_testing_sequences(embeded_emails):



def get_training_batches(embeded_emails,Y,batch_size):

    packed_batches_and_labels = []

    for i in range(0,len(embeded_emails),batch_size):

        email_batch = embeded_emails[i:i+batch_size]
        Y_tmp = Y[i:i+batch_size]

        email_batch_sorted , seq_lengths = sort_and_get_seq_lens(email_batch,Y_tmp)

        packed_batches_and_labels.append( zero_padd_and_pack_batch(email_batch_sorted,seq_lengths) )

    return packed_batches_and_labels



def sort_and_get_seq_lens(email_batch,Y):
    # Sort your batch from largest sequence to the smallest (EACH TUPLE OF THE LIST CONTAINS THE EMAIL BATCH AND THE CORRESPONDING LABELS)

    seq_lengths = np.array( list( map(lambda x: len(x) , email_batch ) ) )
    #sorting in descending order
    email_batch_sorted = [ ( email_batch[ind],Y[ind] ) for ind in np.flip(np.argsort(seq_lengths))]
    #sorting in ascending order
    #email_batch_sorted = [ ( email_batch[ind],Y[ind] ) for ind in np.argsort(seq_lengths) ]

    return email_batch_sorted,seq_lengths



def zero_padd_and_pack_batch(email_batch_sorted,seq_lengths):

    #zero padding until the largest sequence
    email_batch_sorted_and_padded = [email_batch_sorted[0][0]]
    labels_batch = [email_batch_sorted[0][1]]
    for i in range(1,len(email_batch_sorted)):
        padded_arr = np.zeros( ( seq_lengths.max(),email_batch_sorted[0][0].shape[1] ) )
        padded_arr[:email_batch_sorted[i][0].shape[0],:email_batch_sorted[i][0].shape[1]] = email_batch_sorted[i][0]
        email_batch_sorted_and_padded.append( padded_arr )
        labels_batch.append(email_batch_sorted[i][1])
        #email_batch_sorted_and_padded.append( np.expand_dims(padded_arr, axis=1) ) #adding a siglenton dim for the batch size 
    
    #email_batch_sorted_and_padded = np.moveaxis(np.array(email_batch_sorted_and_padded), 0, 1)
    input_batch = torch.from_numpy( np.moveaxis(np.array(email_batch_sorted_and_padded) , 0, 1).copy() ).float()

    #sorting the seq length in descending order--
    sorted_seq_lengths = np.flip(np.sort(seq_lengths)).copy()
    #sorted_seq_lengths = np.sort(seq_lengths)

    pack = torch.nn.utils.rnn.pack_padded_sequence(input_batch, sorted_seq_lengths, batch_first=False, enforce_sorted=True)

    labels = torch.tensor(np.array([[labels_batch]])).permute([0,2,1]).float()

    return pack ,labels

#----------------------------------------------------------------------------------------

def train_and_save_w2v_model(X):
    #A-> array of strings (emails)
    #train a word2vec model with the email dataset

    #s_time = time.time()
    print("Model Training started")
    model = Word2Vec(sentences=X, vector_size=300, window=6, min_count=1, workers=5)
    model.save("word2vec.model")
    #cal_elapsed_time(s_time)

def generate_and_save_trainable_dataset(Path,word2vec_flag):

    try:

        X,Y = load_data(Path)
        if word2vec_flag:
            try:
                w2v_model = Word2Vec.load("word2vec.model")
            except:
                #There is no custom pretrained w2v model so train one (its relativly fast) 
                train_and_save_w2v_model(X)
                w2v_model = Word2Vec.load("word2vec.model")

            model_type = 'custom'
        else:
            w2v_model = KeyedVectors.load_word2vec_format('/GoogleNews-vectors-negative300.bin', binary=True)
            model_type = "pretrained"


        embeded_emails , inds_removed = get_embeded_emails(w2v_model,model_type,X)

        if not(inds_removed.size == 0):
            Y = np.delete( Y , inds_removed )

        
        X_train, X_test, Y_train, Y_test = train_test_split(embeded_emails, Y, test_size=0.35, random_state=1)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)

        #find the max number (in a range) that divides perfect the number of examples-------
        for d in range(2,150+1):
            if not( ( len(X_train) % d ) ) and ( d>30 and d<=200) :
                batch_size = d
            
        try:
            print(batch_size)
        except:
            batch_size = 1
            print("Didnt find any apropriate batch_size...using batch_size=1 ")

        dataset_and_batch_size = {
            "train" : ( X_train,Y_train ),
            "valid" : ( X_val,Y_val ),
            "test"  : ( X_test , Y_test ),
            "batch_size" : batch_size
        }
        

        #SAVE DATASET-----------------------
        with open('dataset_and_batch_size_dict.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(dataset_and_batch_size, f)

        return dataset_and_batch_size

    except: 
        try:
            # Getting back the objects:
            with open('dataset_and_batch_size_dict.pkl' , 'rb') as f:  # Python 3: open(..., 'rb')
                #dataset_dict = pickle.load(f)
                return pickle.load(f)
        except: 
            print("ERROR")            

    





'''
def get_embeded_emails_sorted(model_obj,model_type,X,Y):
    #CREATING THE DATASET
    #CREATING the word embedings for each email-------------------------------------------
    embeded_emails = []
    if model_type=='pretrained':
        for email in X:
            tmp_embeded_email = np.array( list( map( lambda x: get_word2vec_pre_trained(model_obj,x) , email ) ) )
            nb_words_per_email = tmp_embeded_email.shape[0]
            embeded_emails.append( tmp_embeded_email )
    else:
        for email in X:
            tmp_embeded_email = np.array( list( map( lambda x: model_obj.wv[x] , email ) ) )
            nb_words_per_email = tmp_embeded_email.shape[0]
            embeded_emails.append( tmp_embeded_email )
    #----------------------------------------------------------------------------------
    #calculating the max email length and 
    seq_lengths = np.array( list( map(lambda x: len(x), X ) ) )
    max_seq_len = seq_lengths.max()
    # sorting the email list from the largest seq to the smallest
    #embeded_emails_sorted contains a list of tuples (email_embedings,label)
    embeded_emails_sorted = [( embeded_emails[i] , Y[i] ) for i in np.flip(np.argsort(seq_lengths))]
    
    return embeded_emails_sorted
'''