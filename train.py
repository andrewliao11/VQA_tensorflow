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
max_words_q = 30
num_answer = 1000

def get_data():

    dataset = {}
    train_data = {}
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
	img_feature = np.array(tem)
        #dataset['fv_im'] = np.array(tem)
    # load h5 file
    print('Loading h5 file...')
    with h5py.File(h5_data_path,'r') as hf:
	# total number of training data is 215375
	# question is (26, )
	tem = hf.get('ques_train')
	train_data['question'] = np.array(tem)
	# max length is 23
	tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
	# total 82460 img 
	# -----1~82460-----
	tem = hf.get('img_pos_train')
        train_data['img_list'] = np.array(tem)
	# answer is 1~1000
	tem = hf.get('answers')
        train_data['answers'] = np.array(tem)
   
    print('Normalizing image feature')
    if normalize:
	tem =  np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
	img_feature = np.divide(img_feature, np.tile(tem,(1,4096)))

    return dataset, img_feature, train_data


class Answer_Generator():
    def __init__(self, dim_image, n_words_q, dim_hidden, batch_size, drop_out_rate, bias_init_vector=None):
        print('Initialize the model')
	self.dim_image = dim_image
        self.n_words_q = n_words_q
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.drop_out_rate = drop_out_rate

	# 2 layers LSTM cell
	self.lstm = rnn_cell.BasicLSTMCell(self.dim_hidden,self.dim_hidden)
	self.lstm_dropout = rnn_cell.DropoutWrapper(self.lstm,output_keep_prob=1 - self.drop_out_rate)
	self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout] * 2)
	
	# image feature embedded
	self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1,0.1), name='embed_image_W')
	if bias_init_vector is not None:
            self.embed_image_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_image_b')
        else:
            self.embed_image_b = tf.Variable(tf.zeros([dim_hidden]), name='embed_image_b')
	
	# embed the word into lower space
	with tf.device("/cpu:0"):
            self.question_emb_W = tf.Variable(tf.random_uniform([n_words_q, dim_hidden], -0.1, 0.1), name='question_emb_W')

	# embed lower space into answer
	self.answer_emb_W = tf.Variable(tf.random_uniform([dim_hidden, num_answer], -0.1, 0.1), name='answer_emb_W')	
	if bias_init_vector is not None:
            self.answer_emb_b = tf.Variable(bias_init_vector.astype(np.float32), name='answer_emb_b')
        else:
            self.answer_emb_b = tf.Variable(tf.zeros([num_answer]), name='answer_emb_b')	

	# record the final word index
	# question_mask = tf.Variable(tf.zeros([self.batch_size]),name='question_mask')
	

    def build_model(self):
	
	print('building model')
	# placeholder is for feeding data
	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
	question = tf.placeholder(tf.int32, [self.batch_size, max_words_q])
	question_length = tf.placeholder(tf.int32, [self.batch_size])
	label = tf.placeholder(tf.int32, [self.batch_size]) # (batch_size, )
	#question_mask = tf.placeholder(tf.float32, [self.batch_size])	

	# [image] embed image feature to dim_hidden
        image_emb = tf.nn.xw_plus_b(image, self.embed_image_W, self.embed_image_b) # (batch_size, dim_hidden)
        image_emb = tf.nn.dropout(image_emb, self.drop_out_rate)
        image_emb = tf.tanh(image_emb)

	# [answer] ground truth
        labels = tf.expand_dims(label, 1) # (batch_size, 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
        concated = tf.concat(1, [indices, labels]) # (batch_size, 2)
        answer = tf.sparse_to_dense(concated, tf.pack([self.batch_size, num_answer]), 1.0, 0.0) # (batch_size, num_answer)

	# [question_mask] 
	labels_q = tf.expand_dims(question_length, 1) # b x 1
        indices_q = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
        concated_q = tf.concat(1, [indices_q, labels_q]) # b x 2
        question_mask = tf.sparse_to_dense(concated_q, tf.pack([self.batch_size, max_words_q]), 1.0, 0.0) # (batch_size, max_words_q)
	
	probs = []
        loss = 0.0

        state = tf.zeros([self.batch_size, max_words_q]) 
	output = tf.zeros([self.batch_size, max_words_q])

	states = []
	outputs = []
	for j in range(max_words_q): 
            if j == 0:
                question_emb = tf.zeros([self.batch_size, self.dim_hidden])
            else:
                with tf.device("/cpu:0"):
		    tf.get_variable_scope().reuse_variables()
                    question_emb = tf.nn.embedding_lookup(self.question_emb_W, question[:,j-1])
	    output, state = self.stacked_lstm(tf.concat(1,[image_emb, question_emb]), state)
	    # record the state an output
	    states.append(state)
	    outputs.append(output)
	
	
	# predict   
	# pack -> convert input into an array
	outputs = tf.pack(outputs) # (batch_size, max_words_q, dim_hidden)
	output_final = tf.reduce_sum(tf.mul(outputs, question_mask), 1) # (batch_size, )
	answer_pred = tf.nn.xw_plus_b(output_final, self.answer_emb_W, self.answer_emb_b) # (batch_size, num_answer)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(answer_pred, answer) # (batch_size, )
	loss = tf.reduce_sum(cross_entropy)
	loss = loss/self.batch_size

	return loss

## Train Parameter
dim_image = 4096
dim_hidden = 1024
n_epochs = 300
batch_size = 10
learning_rate = 0.0001 #0.001

def train():

    print('Start to load data!')
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())
    
    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
	    dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0.5,
            bias_init_vector = None)
    
    tf_loss = model.build_model()
    
    sess = tf.InteractiveSession()
    writer = tf.train.SummaryWriter('/tmp/tf_log', sess.graph_def)
    saver = tf.train.Saver(max_to_keep=100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()
    
    tStart_total = time.time()
    for epoch in range(n_epochs):
	# shuffle the training data
	index = np.arange(num_train)
        np.random.shuffle(index)
        train_data['question'] = train_data['question'][index,:]
	train_data['length_q'] = train_data['length_q'][index]
	train_data['answers'] = train_data['answers'][index]
        train_data['img_list'] = train_data['img_list'][index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(num_train)
	for current_batch_file_idx in xrange(num_train):

            tStart = time.time()
            # set data into current*
	    current_question = train_data['question'][current_batch_file_idx,:]
            current_length_q = train_data['length_q'][current_batch_file_idx]
            current_answers = train_data['answers'][current_batch_file_idx]
            current_img_list = train_data['img_list'][current_batch_file_idx]
	    # init the para
	    current_img = np.zeros((batch_size, dim_image))
	    current_img_idx = np.zeros((batch_size))
            #current_question = np.zeros((batch_size, max_words_q))
            #current_question_length = np.zeros((batch_size))
            #current_label = np.zeros((batch_size))

            # one batch at a time
            for idx_batch in xrange(batch_size):
                current_img_idx[idx_batch] = current_img_list[idx_batch] # (batch_size, )
		# minus 1 since in MSCOCO the idx is 0~82459
		# 		   VQA 	  the idx is 1~82460	
		current_img[idx_batch,:] = img_feature[current_img_idx-1] # (batch_size, dim_image)
		'''
                idx = np.where(current_batch['label'][:,ind] != -1)[0]
                if len(idx) == 0:
                        continue
                idy = np.where(current_batch['label'][:,ind] == 1)[0]
                if len(idy) == 0:
                        continue
                current_HLness[ind,idx] = current_batch['label'][idx,ind]
                current_HLness_masks[ind,idx] = 1
                current_video_masks[ind,idy[-1]] = 1   	
		'''
	    '''
	    current_captions = current_batch['title']
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=15-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))
	    
	    for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
	    '''
            # do the training process!!!
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
                        tf_question_length: current_length_q,
                        tf_label: current_answers,
                        })
	    loss_epoch[current_batch_file_idx] = loss
            tStop = time.time()
            print ("Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")


if __name__ == '__main__':
    train()
