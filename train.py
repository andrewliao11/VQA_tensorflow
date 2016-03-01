from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import cv2
from tensorflow.models.rnn import rnn_cell
from keras.preprocessing import sequence
from sklearn.metrics import average_precision_score
import json

# path
json_data_path= '/home/andrewliao11/VQA_LSTM_CNN/data_prepro.json'
h5_data_path = '/home/andrewliao11/VQA_LSTM_CNN/data_prepro.h5'
image_feature_path = '/home/andrewliao11/VQA_LSTM_CNN/data_img.h5'

## Some args
normalize = True
def get_data():

    dataset = {}

    # load json file
    print('Loading json file...')
    with open(json_data_path) as data_file:
	data = json.load(data_file)
    for key in data.keys():
	dataset[key] = data[key]
    
    # load image feature
    print('Loading image feature...')
    with h5py.File(image_feature_path,'r') as hf:
	# -----0~82459------
        tem = hf.get('images_train')
        dataset['fv_im'] = np.array(tem)
    # load h5 file
    print('Loading h5 file...')
    with h5py.File(h5_data_path,'r') as hf:
	# total number of training data is 215375
	# question is (26, )
	tem = hf.get('ques_train')
	dataset['question'] = np.array(tem)
	# max length is 23
	tem = hf.get('ques_length_train')
        dataset['lengths_q'] = np.array(tem)
	# total 82460 img 
	# -----1~82460-----
	tem = hf.get('img_pos_train')
        dataset['img_list'] = np.array(tem)
	# answer is 1~1000
	tem = hf.get('answers')
        dataset['answers'] = np.array(tem)
   
    #print(np.multiply(dataset['fv_im'],dataset['fv_im']).shape)
    print('Normalizing image feature')
    if normalize:
	tem =  np.sqrt(np.sum(np.multiply(dataset['fv_im'],dataset['fv_im'])))
	dataset['fv_im'] = np.divide(dataset['fv_im'],np.tile(tem,(1,4096)))
	#local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2))
        #dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,4096)):float()

    return dataset

class Answer_Generator():
    def __init__(self, dim_image, n_words_q, n_words_c, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words_q = n_words_q
	self.n_words_c = n_words_c
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

	# 2 layers LSTM cell
	self.lstm = rnn_cell.BasicLSTMCell(self.dim_hidden,self.dim_hidden)
	self.lstm_dropout = rnn_cell.DropoutWrapper(self.lstm,output_keep_prob=1 - self.drop_out_rate)
	self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout] * 2)

	'''
	self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden,self.dim_hidden,use_peepholes = True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden,2*self.dim_hidden,use_peepholes = True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
	'''
	
	# image feature embedded
	self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1,0.1), name='embed_image_W')
	if bias_init_vector is not None:
            self.embed_image_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_image_b')
        else:
            self.embed_image_b = tf.Variable(tf.zeros([dim_hidden]), name='embed_image_b')
	
	# embed the word into lower space
	with tf.device("/cpu:0"):
            self.question_emb_matrix = tf.Variable(tf.random_uniform([n_words_q, dim_hidden], -0.1, 0.1), name='question_emb_matrix')

	with tf.device("/cpu:0"):
            self.caption_emb_matrix = tf.Variable(tf.random_uniform([n_words_c, dim_hidden], -0.1, 0.1), name='caption_emb_matrix')

	# TODO:embed loser space into word
	'''
	# question embedded
	self.embed_question_W = tf.Variable(tf.random_uniform([dim_hidden, n_words_q], -0.1,0.1), name='embed_question_W')
	if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_question_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_question_b')

	#caption embedded
	with tf.device("/cpu:0"):
            self.caption_emb = tf.Variable(tf.random_uniform([n_words_c, dim_hidden], -0.1, 0.1), name='caption_emb')
	self.embed_caption_W = tf.Variable(tf.random_uniform([dim_hidden, n_words_c], -0.1,0.1), name='embed_caption_W')
	if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
	'''

    def build_model(self):
	# placeholder is for feeding data
	image_feature = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
	# TODO
	#caption_emb = tf.placeholder(tf.int32, [self.batch_size, ??])       
	#question_emb = tf.placeholder(tf.int32, [self.batch_size, ??])

	image_emb = tf.nn.xw_plus_b(image_feature, self.embed_image_W, self.embed_image_b) # [batch_size, dim_hidden]
	
	# initialize LSTM state 
	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        state1_return = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2_return = tf.zeros([self.batch_size, self.lstm2.state_size])

	probs = []
        loss = 0.0
        state1_temp=[]
        state2_temp=[]

	# TODO: calculate max length of caption and question
	for j in range(15): 
            if j == 0:
                question_emb = tf.zeros([self.batch_size, self.dim_hidden])
		caption_emb = tf.zers([self.batch_size, self.dim_hidden])
	    '''
            else:
                # cuttent question/caption embed => ground truth
                with tf.device("/cpu:0"):
		    # TODO: where to feed the data??
		    #current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,j-1])
                    question_emb = tf.nn.embedding_lookup(self.question_emb_matrix, ?)
		    caption_emb = tf.nn.embedding_loofup(self.caption_emb_matrix, ?)
	    '''
	    if j == 0:
		tf.get_variable_scope().reuse_variables()
	    #output, state = self.stacked_lstm(tf.concat(1,[current_embed, output1]), state)
	    '''
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( padding, state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat(1,[current_embed, output1]), state2 ) 
	    '''
    '''
    def build_generator(self):
	# placeholder is for feeding data
	image_feature_emb = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
	# TODO
        #caption_emb = tf.placeholder(tf.int32, [1, ??])
        #question_emb = tf.placeholder(tf.int32, [1, ??])

	# LSTM state
        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        state1_return = tf.zeros([1, self.lstm1.state_size])
        state2_return = tf.zeros([1, self.lstm2.state_size])
    '''
## Train Parameter
dim_image = 4096
dim_hidden = 1024
#embedded_size_question = 1024
#embedded_size_image = 1024
#embedded_size_caption = 1024
# TODO: modify it
n_frame_step = 110

n_epochs = 300
batch_size = 10
learning_rate = 0.0001 #0.001
'''
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector
'''
def train():

    print('Start to load data!')
    dataset = get_data()
    
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())
    print(vocabulary_size_q)
    
    
    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
	    # TODO: calcualte caption vocab size
	    n_words_c = 1000,
	    dim_hidden = dim_hidden,
            #embedded_size_question = embedded_size_question,
	    #embedded_size_image = embedded_size_image,
	    #embedded_size_caption = embedded_size_caption,
            batch_size = batch_size,
            n_lstm_steps = n_frame_step,
            drop_out_rate = 0.5,
            bias_init_vector = None)
    

if __name__ == '__main__':
    train()
