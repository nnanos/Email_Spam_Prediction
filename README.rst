=======================================================================
Email Spam detection with Deep Learning
=======================================================================

Description
============

In this repository I do some experiments on email spam detection. 
The dataset that is being used is the spam_or_not_spam.csv file which contains two columns.
The first column includes the text from various emails while the second column informs us whether they were spam or not: value 1 for spam, 0 otherwise.
My goal is to try to guess the information in the second column using a neural network.
To approach this problem I am using Deep Learning and more specifically RNNs and LSTMs.


Module Description 
============


* Load_data_functions.py

In this module there are functions required to load the dataset and
preprocessing of it. The only pre-processing performed is that of conversion
of the words of each email in word embeddings. For the latter, two capabilities have been implemented
, either training a word2vec model on the existing dataset or
using the pre-trained google model (GoogleNews-vectors-negative300). The approach followed for the experiments was the first because:
1) I saw satisfactorily NN results and
2) because loading the google model into memory is costly as it is trained on a large volume of words (it is 3Gb) .The implemented functions simulate the following steps (which are executed
serially):
(*Note that everything is in memory because the dataset and word embeddings are
small)

	#.  **load_data()**
		Loads the given csv file into a dataframe. It then separates the emails from the
		respective labels and finally returns 2 lists. Each element of the first is an np
		array that contains the words (strs) of an email and the second contains the corresponding labels.

	#.  **get_embeded_emails()**
		Converts each email (sequence of strs) to a sequence of vectors (word
		embeddings). Now a list is returned, each element of which is a matrix of
		which represents an email (each vector is a word) .

	#.  **get_training_batches()**
		For each mini-batch of embedded emails, the emails are sorted by
		largest to smallest sequence , then zero padding all emails to
		length of the longest email of the mini-batch (because when training an RNN
		each batch sequence needs to be the same length (for efficient implementation
		of as MM multiplication) ) . So what is returned is a list of mini-batches
		each of which contains sequences (embedded emails) of the same length
		(we used packing).


* Model.py

The Model class represents the architecture of the network. Two possibilities were implemented,
a simple elman RNN and an LSTM which can be selected from input arguments.
In each case the output of the state of the last step of the recursive network
enters a linear level which implements an inner product.
Finally the neural net is trained ( evaluation mode) the response of the linear level which is a scalar becomes an input to a sigmoid non-linearity in order to model the probability that the email that processed the neural net to be spam or not. Whereas when the network is in training mode then
again the probability is modeled in the same way as before but now we don't have to pass
the response of the linear level from the sigmoid because this is done in the calculation of the
loss function ( BCEWithLogitsLoss() ) of pytorch. 

	* IMAGE


The image above shows details of the architecture of the implemented NNs
and specifically of vanilla RNN although LSTM follows the same mentality simply )
the cell becomes more complex so as to deal with i.e vanishing gradient problems
for long time series. I also list the objective used for
minimization.


* train.py

This module was implemented in order to train a recurrent neural network. The
basic logic is described in the following steps:

	#. Loading the data using the functions of the Load_data_functions.py module.

	#. Initialization of objects : Model,critireon,optimizer,scheduler

	#.


* Evaluation.py


Experiments
=============






============

Free software: MIT license
============

