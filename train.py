'''
VisualQA:

[baseline]
Method:
treat image as the first question word as input 
and take the last "output" as the answer predicted
Architecture:
2-layer LSTM 

1 batch(125): 0.84s
'''
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
import pdb

# path
json_data_path= '/home/andrewliao11/Work/VQA_challenge/data_prepro.json'
h5_data_path = '/home/andrewliao11/Work/VQA_challenge/data_prepro.h5'
image_feature_path = '/home/andrewliao11/Work/VQA_challenge/data_img.h5'


## Some args
normalize = True
max_words_q = 26
num_answer = 1000

# Check point
save_checkpoint_every = 25000           # how often to save a model checkpoint?
model_path = '/home/andrewliao11/Work/VQA_challenge/model/'

## Train Parameter
dim_image = 4096
dim_hidden = 512
n_epochs = 30
batch_size = 125
learning_rate = 0.0001 #0.001


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
	# convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
	# answer is 1~1000
	# change to 0~999
	tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1
   
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
	self.lstm1 = rnn_cell.LSTMCell(2*self.dim_hidden,2*self.dim_hidden,use_peepholes = True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2 = rnn_cell.LSTMCell(2*self.dim_hidden,2*self.dim_hidden,use_peepholes = True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
	'''
	self.lstm = rnn_cell.BasicLSTMCell(self.dim_hidden)
	#self.lstm_dropout = rnn_cell.DropoutWrapper(self.lstm,output_keep_prob=1 - self.drop_out_rate)
	self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm] * 2)
	'''
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
	self.answer_emb_W = tf.Variable(tf.random_uniform([2*dim_hidden, num_answer], -0.1, 0.1), name='answer_emb_W')	
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
	question_mask = tf.placeholder(tf.int32, [max_words_q, self.batch_size, 2*self.dim_hidden])
	label = tf.placeholder(tf.int32, [self.batch_size, num_answer]) # (batch_size, )
	label = tf.to_float(label)

	# [image] embed image feature to dim_hidden
        image_emb = tf.nn.xw_plus_b(image, self.embed_image_W, self.embed_image_b) # (batch_size, dim_hidden)
        image_emb = tf.nn.dropout(image_emb, self.drop_out_rate)
        image_emb = tf.tanh(image_emb)
	
	probs = []
        loss = 0.0

	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
	states = []
	outputs = []
	for j in range(max_words_q): 
            if j == 0:
                question_emb = tf.zeros([self.batch_size, self.dim_hidden])
            else:
                with tf.device("/cpu:0"):
		    tf.get_variable_scope().reuse_variables()
                    question_emb = tf.nn.embedding_lookup(self.question_emb_W, question[:,j-1])

	    with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout(tf.concat(1,[image_emb, question_emb]), state1 )
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout(output1, state2 )

	    # record the state an output
	    states.append(state2)
	    outputs.append(output2)
	    	
	
	# predict   
	# pack -> convert input into an array
	output = tf.pack(outputs) # (max_words_q, batch_size, 4*dim_hidden)
	output_final = tf.reduce_sum(tf.mul(output, tf.to_float(question_mask)), 0) # (batch_size, 2*dim_hidden)
	answer_pred = tf.nn.xw_plus_b(output_final, self.answer_emb_W, self.answer_emb_b) # (batch_size, num_answer)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(answer_pred, label) # (batch_size, )
	loss = tf.reduce_sum(cross_entropy)
	loss = loss/self.batch_size

	return loss, image, question, question_mask, question_length, label

def train():

    print('Start to load data!')
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())
    
    # answers = 2 ==> [0,0,1,0....,0]    
    answers = np.zeros([num_train, num_answer])
    answers[range(num_train),np.expand_dims(train_data['answers'],1)[:,0]] = 1 # all x num_answers 
    

    #pdb.set_trace()
    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
	    dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0.5,
            bias_init_vector = None)
    
    tf_loss, tf_image, tf_question, tf_question_mask, tf_question_length, tf_label = model.build_model()
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    writer = tf.train.SummaryWriter('/home/andrewliao11/Word/VQA_challenge/tmp/tf_log', sess.graph_def)
    saver = tf.train.Saver(max_to_keep=100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()
    
    tStart_total = time.time()
    for epoch in range(n_epochs):
	# shuffle the training data
	index = np.arange(num_train)
        np.random.shuffle(index)
        train_data['question'] = train_data['question'][index,:] # (num_train, max_words_q)
	train_data['length_q'] = train_data['length_q'][index] # (num_train, )
	answers = answers[index,:] # num_train x num_answers
        train_data['img_list'] = train_data['img_list'][index]
	
        tStart_epoch = time.time()
        loss_epoch = np.zeros(num_train)

	for current_batch_start_idx in xrange(0,num_train-1,batch_size):
            tStart = time.time()
            # set data into current*
	    if current_batch_start_idx + batch_size < num_train:
		current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
	    else:
		current_batch_file_idx = range(current_batch_start_idx,num_train)
	    current_question = np.zeros([batch_size, max_words_q])
            current_length_q = np.zeros(batch_size)
            current_answers = np.zeros([batch_size, num_answer])
            current_img_list = np.zeros(batch_size)
	    current_question = train_data['question'][current_batch_file_idx,:]
            current_length_q = train_data['length_q'][current_batch_file_idx]
            current_answers = answers[current_batch_file_idx,:]
            current_img_list = train_data['img_list'][current_batch_file_idx]
	    current_img = np.zeros((batch_size, dim_image))
	    current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
	   
	    current_question_mask = np.zeros([max_words_q, batch_size, 2*dim_hidden])
    	    current_question_mask[current_length_q, range(batch_size), :] = 1 #(max_words_q, batch_size, 2*dim_hidden)

            # do the training process!!!
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
                        tf_question_length: current_length_q,
			tf_question_mask: current_question_mask,
                        tf_label: current_answers
                        })
	    loss_epoch[current_batch_file_idx] = loss
            tStop = time.time()
            print ("Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")

	
	# every 10 epoch: print result
	if np.mod(epoch, 10) == 0:
            print ("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
	print ("Epoch:", epoch, " done. Loss:", np.mean(loss_epoch))
        tStop_epoch = time.time()
        print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
    
    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")	


if __name__ == '__main__':
    with tf.device('/gpu:'+str(10)):
        train()
