'''
VisualQA:

modified:
reduce redundant matrix

Architecture:
 2-layer LSTM
+attention map
+img feature
=answer generation

TITAN X
1 batch(125): 0.36s
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
from scipy import io


## Train Parameter
n_epochs = 100
batch_size = 125
TRAIN = True
TEST = False
normalize = True
gpu_id = str(0)
model_name = 'ABC_LSTM_lr_v1'

# Some path
json_data_path= '/home/andrewliao11/Work/VQA_challenge/data_prepro.json'
h5_data_path = '/home/andrewliao11/Work/VQA_challenge/data_prepro.h5'
image_feature_path = '/home/andrewliao11/Work/VQA_challenge/data_img.h5'
train_fcn_path = '/data/andrewliao11/FCN/train2014'
#test_fcn_path = '/media/VSlab3/andrewliao11/FCN/fc7/val2014'
model_test = 'models/'+model_name+'/model-100' # default value
meta_path = '/home/andrewliao11/Work/VQA_challenge/meta/'+model_name
model_path = '/home/andrewliao11/Work/VQA_challenge/models/'+model_name+'/'
writer_path = '/home/andrewliao11/Work/VQA_challenge/tmp/'+model_name

## Some args
max_words_q = 26
num_answer = 1000
dim_image = 4096
dim_hidden = 512
word_emb = 200
start_lr_rate = 0.01 # 0.0001,0.002,
kernel_h = 2
kernel_w = 2
att_c =5
save_checkpoint_every = 25000           # how often to save a model checkpoint?
decay_e = 1723		# decay every 100000 updates
decay_rate = 0.9	# decay 0.95 times


def get_train_data():

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

def get_test_data():
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(json_data_path) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(image_feature_path,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_test')
        img_feature = np.array(tem)
        #dataset['fv_im'] = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(h5_data_path,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        # -----1~82460-----
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        # MC_answer_test
        tem = hf.get('MC_ans_test')
        test_data['MC_ans_test'] = np.array(tem)
    return dataset, img_feature, test_data


def get_fcn(path,file_list):

    print('Loading batch fcn')
    fcns = []
    # file_list = ( u'train2014/COCO_train2014_000000516782.jpg',....)
    for img in file_list:
	img = img.encode()
	img = img.split('/')[1]
	img = io.loadmat(path+'/'+img+'_blob_0.mat')
	# fcn shape is 13,13,1024,1
	fcn = np.reshape(img['data'],[13,13,1024])
	fcns.append(fcn)

    fcns = np.asarray(fcns)
    return fcns

class Answer_Generator():
    def __init__(self, dim_image, n_words_q, dim_hidden, kernel_w, kernel_h, batch_size, drop_out_rate, bias_init_vector=None):
        print('Initialize the model')
	self.dim_image = dim_image
        self.n_words_q = n_words_q
        self.dim_hidden = dim_hidden
	self.kernel_w = kernel_w
	self.kernel_h = kernel_h
        self.batch_size = batch_size
        self.drop_out_rate = drop_out_rate

	# 2 layers LSTM cell
	self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden, input_size=word_emb, use_peepholes = True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden, input_size=self.dim_hidden, use_peepholes = True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
	
	###
	# embed feature to answer space	
	# [image]
	self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, num_answer], -0.1,0.1), name='embed_image_W')
	# [sentence]                                                                                                                          
        self.emb_sentence_W = tf.Variable(tf.random_uniform([self.dim_hidden, num_answer], -0.1, 0.1), name='embed_sentence_W')
	# [attention map]
        self.emb_att_W = tf.Variable(tf.random_uniform([13*13*att_c, num_answer], -0.1, 0.1), name='embed_att_W')

	# [question]
        self.question_emb_W = tf.Variable(tf.random_uniform([n_words_q, word_emb], -0.1, 0.1), name='question_emb_W')

	# embed lower space into answer
	if bias_init_vector is not None:
            self.answer_emb_b = tf.Variable(bias_init_vector.astype(np.float32), name='answer_emb_b')
        else:
            self.answer_emb_b = tf.Variable(tf.zeros([num_answer]), name='answer_emb_b')	

	# kernel emb
	self.kernel_emb_W = tf.Variable(tf.random_uniform([dim_hidden, self.kernel_h*self.kernel_w*1024*att_c], -0.1, 0.1), name='kerenl_emb_W')	

    def build_model(self):
	
	print('building model')
	# placeholder is for feeding data
	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
	question = tf.placeholder(tf.int32, [self.batch_size, max_words_q])
	question_mask = tf.placeholder(tf.int32, [max_words_q, self.batch_size, 2*self.dim_hidden])
	label = tf.placeholder(tf.int32, [self.batch_size, num_answer]) # (batch_size, )
	label = tf.to_float(label)
	fcn = tf.placeholder(tf.float32, [self.batch_size, 13, 13, 1024])

        loss = 0.0

	#-----------------------Question Understanding--------------------------
	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
	states = []
	outputs = []
	for j in range(max_words_q): 
            if j == 0:
                question_emb = tf.zeros([self.batch_size, word_emb])
            else:
                with tf.device("/cpu:0"):
		    tf.get_variable_scope().reuse_variables()
                    question_emb = tf.nn.embedding_lookup(self.question_emb_W, question[:,j-1])
	    with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout(question_emb, state1)
	    with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout(output1, state2 )
	    # record the state an output
	    states.append(state2)
	    outputs.append(output2)
	    	
	# pack -> convert input into an array
	states = tf.pack(states) # (max_words_q, batch_size, 2*dim_hidden)
        state = tf.reduce_sum(tf.mul(states, tf.to_float(question_mask)), 0) # (batch_size, 2*dim_hidden)
        state = tf.slice(state, [0,0], [self.batch_size, dim_hidden]) # (batch_size, dim_hidden)
	
	#------------------------Attention Map-------------------------------
	kernel = tf.matmul(state, self.kernel_emb_W) # (batch_size, kernel_w*kernel_h*1024*att_c)
	kernel = tf.reshape(kernel,[self.batch_size, self.kernel_h, self.kernel_w, 1024, att_c])
	kernel = tf.nn.dropout(kernel, 1-self.drop_out_rate) # (batch_size, kernel_h, kernel_w, 1024, att_c)
	kenel = tf.tanh(kernel)
	# conv2d
	att_maps = []
	for j in range(self.batch_size):
	    att_map = tf.nn.conv2d(tf.reshape(fcn[j,:,:,:],[1,13,13,1024]), kernel[j,:,:,:,:], [1,1,1,1], padding='SAME')
	    att_maps.append(att_map)
	att_maps = tf.pack(att_maps) # (batch_size, 13,13,att_c)
	att_flatten = tf.reshape(att_maps,[self.batch_size,-1]) # flatten the vector (batch_size, 13*13*att_c)

	#------------------------Answer Generation----------------------------
	# [image] embed image feature to dim_hidden
        image_emb = tf.matmul(image, self.embed_image_W) # (batch_size, dim_hidden)
        image_emb = tf.nn.dropout(image_emb, 1-self.drop_out_rate)
	image_emb = tf.tanh(image_emb)
	# [sentence]
	sentence_emb = tf.matmul(state, self.emb_sentence_W)
	sentence_emb = tf.nn.dropout(sentence_emb, 1-self.drop_out_rate)
	sentence_emb = tf.tanh(sentence_emb)
	# [attention_map]
	att_emb = tf.matmul(att_flatten, self.emb_att_W) # (batch_size, dim_hidden)
	att_emb = tf.nn.dropout(att_emb, 1-self.drop_out_rate)
	att_emb = tf.tanh(att_emb)


	# answer pred -> tanh(Wi*I+Ws*S+Wa*A+b)
	answer_pred = tf.add(image_emb, sentence_emb)
	answer_pred = tf.add(answer_pred, att_emb)
	answer_pred = tf.add(answer_pred, self.answer_emb_b)
	answer_pred = tf.nn.dropout(answer_pred, 1-self.drop_out_rate)
	answer_pred = tf.tanh(answer_pred)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(answer_pred, label) # (batch_size, )
	loss = tf.reduce_sum(cross_entropy)
	loss = loss/self.batch_size

	return loss, image, fcn, question, question_mask, label

    def build_generator(self):

    	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
    	question = tf.placeholder(tf.int32, [self.batch_size, max_words_q])
   	question_mask = tf.placeholder(tf.int32, [max_words_q, self.batch_size, 2*self.dim_hidden])
	fcn = tf.placeholder(tf.float32, [self.batch_size, 13, 13, 1024])

	#-----------------------Question Understanding--------------------------
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
                output1, state1 = self.lstm1_dropout(question_emb, state1 )
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout(output1, state2 )

            # record the state an output
            states.append(state2)
            outputs.append(output2)

        # predict
        # pack -> convert input into an array
        states = tf.pack(states) # (max_words_q, batch_size, 2*dim_hidden)
        state = tf.reduce_sum(tf.mul(states, tf.to_float(question_mask)), 0) # (batch_size, 2*dim_hidden)
        state = tf.slice(state, [0,0], [self.batch_size, dim_hidden]) # (batch_size, dim_hidden)
  
	#------------------------Attention Map-------------------------------
        kernel = tf.matmul(state, self.kernel_emb_W) # (batch_size, kernel_w*kernel_h*1024)
        kernel = tf.reshape(kernel,[self.batch_size, self.kernel_h, self.kernel_w, 1024, 1])
        kernel = tf.nn.dropout(kernel, 1-self.drop_out_rate) # (batch_size, kernel_h, kernel_w, 1024, 1)
        kenel = tf.tanh(kernel)

	att_maps = []
        for j in range(self.batch_size):
            att_map = tf.nn.conv2d(tf.reshape(fcn[j,:,:,:],[1,13,13,1024]), kernel[j,:,:,:,:], [1,1,1,1], padding='SAME')
            att_maps.append(att_map)
        att_maps = tf.pack(att_maps) # (batch_size, 13*13)
        att_flatten = tf.reshape(att_maps,[self.batch_size,-1]) # flatten the vector (batch_size, 13*13)

	#------------------------Answer Generation----------------------------
        # [image] embed image feature to dim_hidden
        image_emb = tf.matmul(image, self.embed_image_W) # (batch_size, dim_hidden)
        image_emb = tf.nn.dropout(image_emb, 1-self.drop_out_rate)
        # [sentence]
        sentence_emb = tf.matmul(state, self.emb_sentence_W)
        sentence_emb = tf.nn.dropout(sentence_emb, 1-self.drop_out_rate)
        # [attention_map]
        att_emb = tf.matmul(att_flatten, self.emb_att_W) # (batch_size, dim_hidden)
        att_emb = tf.nn.dropout(att_emb, 1-self.drop_out_rate)


	# answer pred -> tanh(Wi*I+Ws*S+Wa*A+b)
        answer_pred = tf.add(image_emb, sentence_emb)
        answer_pred = tf.add(answer_pred, att_emb)
        answer_pred = tf.add(answer_pred, self.answer_emb_b)
        answer_pred = tf.nn.dropout(answer_pred, 1-self.drop_out_rate)
        answer_pred = tf.tanh(answer_pred)
	generated_ans = tf.argmax(answer_pred, 1) # b

    	return generated_ans, image, fcn, question, question_mask

def train():

    print('Start to load data!')
    dataset, img_feature, train_data = get_train_data()
    #fcn = get_train_fcn(dataset['unique_img_train'])
    num_train = train_data['question'].shape[0]
    # count question and caption vocabulary size
    # plus one, since 0 means nothing!
    vocabulary_size_q = len(dataset['ix_to_word'].keys())+1
    
    # answers = 2 ==> [0,0,1,0....,0]    
    answers = np.zeros([num_train, num_answer])
    answers[range(num_train),np.expand_dims(train_data['answers'],1)[:,0]] = 1 # all x num_answers 
    

    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
	    kernel_w = kernel_w,
	    kernel_h = kernel_h,
	    dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0.5,
            bias_init_vector = None)
    
    tf_loss, tf_image, tf_fcn, tf_question, tf_question_mask, tf_label = model.build_model()
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    writer = tf.train.SummaryWriter(writer_path, sess.graph_def)
    saver = tf.train.Saver(max_to_keep=100)
    step = tf.Variable(0, trainable=False)
    # decay every 10 epoch
    learning_rate = tf.train.exponential_decay(start_lr_rate, step, decay_e, decay_rate)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss,global_step = step)
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
	#fcn = fcn[index,:]	

        tStart_epoch = time.time()
        loss_epoch = np.zeros(num_train)
	for current_batch_start_idx in xrange(0,num_train-1,batch_size):
            tStart = time.time()
            # set data into current*
	    if current_batch_start_idx + batch_size < num_train:
		current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
	    else:
		current_batch_file_idx = range(current_batch_start_idx,num_train)
	    # initialize
	    current_question = np.zeros([batch_size, max_words_q])
            current_length_q = np.zeros(batch_size)
            current_answers = np.zeros([batch_size, num_answer])
            current_img_list = np.zeros(batch_size)
	    current_img = np.zeros([batch_size,dim_hidden])
	    current_fcn = np.zeros([batch_size,13,13,1024])
	    # assign value
	    current_question = train_data['question'][current_batch_file_idx,:]
            current_length_q = train_data['length_q'][current_batch_file_idx]
            current_answers = answers[current_batch_file_idx,:]
            current_img_list = train_data['img_list'][current_batch_file_idx] # (batch_size, which picture)
	    current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
	    #current__fcn = fcn[current_img_list,:]
	    current_fcn = get_fcn(train_fcn_path, np.take(dataset['unique_img_train'], current_img_list))
	    current_question_mask = np.zeros([max_words_q, batch_size, 2*dim_hidden])
    	    current_question_mask[current_length_q, range(batch_size), :] = 1 #(max_words_q, batch_size, 2*dim_hidden)

            # do the training process!!!
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
			tf_question_mask: current_question_mask,
                        tf_label: current_answers,
			tf_fcn: current_fcn
                        })
	    loss_epoch[current_batch_file_idx] = loss
            tStop = time.time()
            print ("Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")

	# every 20 epoch: print result
	if np.mod(epoch, 20) == 0:
            print ("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
	    #test(model_path+'model-'+str(epoch))
	np.save(meta_path+'/loss_'+str(epoch),loss_epoch)
	print ("Epoch:", epoch, " done. Loss:", np.mean(loss_epoch))
        tStop_epoch = time.time()
        print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
    
    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")	


def test(m=model_test):

    batch_size = 2
    print ('[TESTing]')
    print('Start to load data!')
    dataset, img_feature, test_data = get_test_data()
    num_test = test_data['question'].shape[0]
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())+1

    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
            kernel_w = kernel_w,
            kernel_h = kernel_h,
            dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0,
            bias_init_vector = None)

    tf_generated_ans, tf_image, tf_fcn, tf_question, tf_question_mask = model.build_generator()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, m)
    tf.initialize_all_variables().run()

    result = []
    tStart_total = time.time()
    for current_batch_start_idx in xrange(0,num_test-1,batch_size):
	tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)
        current_question = np.zeros([batch_size, max_words_q])
        current_length_q = np.zeros(batch_size)
        current_img_list = np.zeros(batch_size)
	current_answers = np.zeros(batch_size)
        current_fcn = np.zeros([batch_size,13,13,1024])
        current_img = np.zeros((batch_size, dim_image))
        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
	current_ques_id = test_data['ques_id'][current_batch_file_idx]
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
	current_fcn = get_fcn(test_fcn_path, np.take(dataset['unique_img_test'], current_img_list))
        current_question_mask = np.zeros([max_words_q, batch_size, 2*dim_hidden])
        current_question_mask[current_length_q, range(batch_size), :] = 1 #(max_words_q, batch_size, 2*dim_hidden)

	# do the testing process!!!
        generated_ans = sess.run(
                tf_generated_ans,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question,
                    tf_question_mask: current_question_mask,
                    tf_fcn: current_fcn
                    })
	for i in xrange(0,batch_size):
            ans = dataset['ix_to_ans'][str(generated_ans[i]+1)]
            if(current_ques_id[i] == 0):
                continue
            result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})
        tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx)
        print ("Time Cost:", round(tStop - tStart,2), "s")
    
    print ("Testing done.")
    # Save to JSON
    print ('Saving result...')
    my_list = list(result)
    dd = json.dump(my_list,open('result_train/'+m.split('/')[1]+'-'+m.split('/')[2]+'.json','w'))
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")

    #return accuracy

if __name__ == '__main__':
    if not os.path.exists(model_path):
    	os.mkdir(model_path)
    if not os.path.exists(meta_path):
    	os.mkdir(meta_path)
    if TRAIN:
    	with tf.device('/gpu:'+gpu_id):
            train()
    if TEST:
    	test()
