'''
VisualQA:

[baseline]
Method:
treat image as the first question word as input
and take the last "state" as the answer predicted

Architecture:
change the LSTM size
2-layer LSTM

1 batch(125): 0.26s
1 epoch: 345s
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


model_name = 'ABC_v6'
# path
json_data_path= '/home/andrewliao11/Work/VQA_challenge/data_prepro.json'
h5_data_path = '/home/andrewliao11/Work/VQA_challenge/data_prepro.h5'
image_feature_path = '/home/andrewliao11/Work/VQA_challenge/data_img.h5'
train_fcn_path = '/home/andrewliao11/train2014'
#train_fcn_path = '/data/andrewliao11/FCN/train2014'
#test_fcn_path = '/media/VSlab3/andrewliao11/FCN/fc7/val2014'
model_test = 'models/'+model_name+'/model-100' # default value
model_path = '/home/andrewliao11/Work/VQA_challenge/models/'+model_name+'/'
writer_path = '/home/andrewliao11/Work/VQA_challenge/tmp/'+model_name

## Some args
TRAIN = True
TEST = False
gpu_id = str(3)
normalize = True
max_words_q = 26
num_answer = 1000
model_name = 'lstm'
kernel_w = 3
kernel_h = 3

# Check point
save_checkpoint_every = 25000           # how often to save a model checkpoint?

## Train Parameter
dim_image = 4096
dim_hidden = 512
n_epochs = 100
batch_size = 125
learning_rate = 0.0001 #0.001



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
        tem = hf.get('images_train')
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
    def __init__(self, dim_image, n_words_q, dim_hidden, batch_size, drop_out_rate, bias_init_vector=None):
        print('Initialize the model')
	self.dim_image = dim_image
        self.n_words_q = n_words_q
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.drop_out_rate = drop_out_rate	
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h


	# 2 layers LSTM cell
	self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden, input_size=self.dim_hidden, use_peepholes = True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden, input_size=self.dim_hidden, use_peepholes = True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
	
	# image feature embedded
	self.embed_image_W = tf.Variable(tf.random_uniform([self.dim_image, self.dim_hidden], -0.1,0.1), name='embed_image_W')
	self.embed_state_W = tf.Variable(tf.random_uniform([self.dim_hidden, self.dim_hidden], -0.1,0.1), name='embed_state_W')
	self.embed_att_W = tf.Variable(tf.random_uniform([13*13*1024, dim_hidden], -0.1, 0.1), name='embed_att_W')

	self.kernel_emb_W = tf.Variable(tf.random_uniform([dim_hidden, kernel_w*kernel_h*1024], -0.1, 0.1), name='kernel_emb_W')
	if bias_init_vector is not None:
            self.kernel_emb_b = tf.Variable(bias_init_vector.astype(np.float32), name='kernel_emb_b')
        else:
            self.kernel_emb_b = tf.Variable(tf.zeros([kernel_w*kernel_h*1024]), name='kernel_emb_b')

	# embed the word into lower space
	with tf.device("/cpu:0"):
            self.question_emb_W = tf.Variable(tf.random_uniform([n_words_q, dim_hidden], -0.1, 0.1), name='question_emb_W')

	if bias_init_vector is not None:
            self.h_emb_b = tf.Variable(bias_init_vector.astype(np.float32), name='h_emb_b')
        else:
            self.h_emb_b = tf.Variable(tf.zeros([self.dim_hidden]), name='h_emb_b')
	# embed lower space into answer
	self.answer_emb_W = tf.Variable(tf.random_uniform([dim_hidden, num_answer], -0.1, 0.1), name='answer_emb_W')	
	if bias_init_vector is not None:
            self.answer_emb_b = tf.Variable(bias_init_vector.astype(np.float32), name='answer_emb_b')
        else:
            self.answer_emb_b = tf.Variable(tf.zeros([num_answer]), name='answer_emb_b')	

	

    def build_model(self):
	
	print('building model')
	# placeholder is for feeding data
	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
	question = tf.placeholder(tf.int32, [self.batch_size, max_words_q])
	question_length = tf.placeholder(tf.int32, [self.batch_size])
	question_mask = tf.placeholder(tf.int32, [max_words_q+1, self.batch_size, 2*self.dim_hidden])
	label = tf.placeholder(tf.int32, [self.batch_size, num_answer]) # (batch_size, )
	label = tf.to_float(label)
	fcn = tf.placeholder(tf.float32, [self.batch_size,13,13,1024])

        loss = 0.0

	#-----------------Question Understanding---------------------------
	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
	states = []
	outputs = []
	for j in range(max_words_q+1): 
            if j == 0:
                question_emb = tf.zeros([self.batch_size, self.dim_hidden])
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
	    		
	# predict   
	# pack -> convert input into an array
	states = tf.pack(states) # (max_words_q+1, batch_size, 2*dim_hidden)
        state = tf.reduce_sum(tf.mul(states, tf.to_float(question_mask)), 0) # (batch_size, 2*dim_hidden)
        state = tf.slice(state, [0,0], [self.batch_size, dim_hidden]) # (batch_size, dim_hidden)
	# average the states
	length = tf.expand_dims(tf.to_float(question_length), 1) # (batch_size, 1)
        length = tf.tile(length, [1,dim_hidden]) # (batch_size, dim_hidden))
        state = tf.div(state, length)
		
	#------------------------Attention Map-------------------------------
        kernel = tf.nn.xw_plus_b(state, self.kernel_emb_W, self.kernel_emb_b) # (batch_size, kernel_w*kernel_h*1024*1)
        kernel = tf.reshape(kernel,[self.batch_size, self.kernel_h, self.kernel_w, 1024, 1])
        #kernel = tf.nn.dropout(kernel, 1-self.drop_out_rate) # (batch_size, kernel_h, kernel_w, 1024, 1)
        kernel = tf.sigmoid(kernel)
        # conv2d
        z_s = []
        for j in range(self.batch_size):
            z = tf.nn.conv2d(tf.reshape(fcn[j,:,:,:],[1,13,13,1024]), kernel[j,:,:,:,:], [1,1,1,1], padding='SAME')
            z_s.append(z)
        z_s = tf.pack(z_s) # (batch_size,1,13,13,1)
        z_s = tf.reshape(z_s, [batch_size,13*13])
        m = tf.nn.softmax(z_s)
        m = tf.reshape(m, [batch_size,13,13])
        # value across 1024 are the same
        m = tf.expand_dims(m,3) # (batch_size,13,13,1)
        m = tf.tile(m, [1,1,1,1024]) # (batch_size,13,13,1024)
        att_maps = tf.mul(m,fcn) # (batch_size, 13,13,1024)
        # TODO add 1x1 conv to avoid overfit
	

	# [image]
	image_emb = tf.matmul(image, self.embed_image_W)
	#image_emb = tf.nn.dropout(image_emb, 1-self.drop_out_rate)
	# [question]
	state_emb = tf.matmul(state, self.embed_state_W)
	#state_emb = tf.nn.dropout(state_emb, 1-self.drop_out_rate)
	# [attention maps]
        # flatten
        att_maps = tf.reshape(att_maps,[self.batch_size,-1]) # flatten the vector (batch_size, 13*13*1024)
	att_maps = tf.matmul(att_maps, self.embed_att_W)
	att_maps = tf.nn.relu(att_maps) # (batch_size, dim_hidden)
	# h -> tanh(Wi*I+Ws*S+Wa*A+b)
        h = tf.add(image_emb, state)
	h = tf.add(h,att_maps)
	h = tf.add(h, self.h_emb_b) # b x dim_hidden
	answer_pred = tf.nn.xw_plus_b(h, self.answer_emb_W, self.answer_emb_b) # (batch_size, num_answer)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(answer_pred, label) # (batch_size, )
	loss = tf.reduce_sum(cross_entropy)
	loss = loss/self.batch_size

	return loss, image, question, question_mask, question_length, label, fcn

    def build_generator(self):

    	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
    	question = tf.placeholder(tf.int32, [self.batch_size, max_words_q])
    	question_length = tf.placeholder(tf.int32, [self.batch_size])
   	question_mask = tf.placeholder(tf.int32, [max_words_q, self.batch_size, 2*self.dim_hidden])

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
        states = tf.pack(states) # (max_words_q, batch_size, 2*dim_hidden)
        state = tf.reduce_sum(tf.mul(states, tf.to_float(question_mask)), 0) # (batch_size, 2*dim_hidden)
        state = tf.slice(state, [0,0], [self.batch_size, dim_hidden]) # (batch_size, dim_hidden)
        answer_pred = tf.nn.xw_plus_b(state, self.answer_emb_W, self.answer_emb_b) # (batch_size, num_answer)
	generated_ans = tf.argmax(answer_pred, 1) # b

    	return generated_ans, image, question, question_mask, question_length

def train():

    print('Start to load data!')
    dataset, img_feature, train_data = get_train_data()
    num_train = train_data['question'].shape[0]
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())+1
    
    # answers = 2 ==> [0,0,1,0....,0]    
    answers = np.zeros([num_train, num_answer])
    answers[range(num_train),np.expand_dims(train_data['answers'],1)[:,0]] = 1 # all x num_answers 
    
    train_data['length_q'] = train_data['length_q']+1

    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
	    dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0.5,
            bias_init_vector = None)
    
    tf_loss, tf_image, tf_question, tf_question_mask, tf_question_length, tf_label, tf_fcn = model.build_model()
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    writer = tf.train.SummaryWriter(writer_path, sess.graph_def)
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
	    current_img = np.zeros([batch_size, dim_image])
	    current_fcn = np.zeros([batch_size,13,13,1024])
	    current_question = train_data['question'][current_batch_file_idx,:]
            current_length_q = train_data['length_q'][current_batch_file_idx]
            current_answers = answers[current_batch_file_idx,:]
            current_img_list = train_data['img_list'][current_batch_file_idx]
	    current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
	    current_fcn = get_fcn(train_fcn_path, np.take(dataset['unique_img_train'], current_img_list))

	    current_question_mask = np.zeros([max_words_q+1, batch_size, 2*dim_hidden])
	    for i in range(batch_size):
                current_question_mask[0:current_length_q[i],i,:] = 1

    	    #current_question_mask[current_length_q, range(batch_size), :] = 1 #(max_words_q, batch_size, 2*dim_hidden)

            # do the training process!!!
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
                        tf_question_length: current_length_q,
			tf_question_mask: current_question_mask,
                        tf_label: current_answers,
			tf_fcn: current_fcn
                        })
	    loss_epoch[current_batch_file_idx] = loss
            tStop = time.time()
            print ("Epoch:", epoch, " Batch:", current_batch_start_idx, " Loss:", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")

	
	# every 20 epoch: print result
	if np.mod(epoch, 20) == 0:
            print ("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
	print ("Epoch:", epoch, " done. Loss:", np.mean(loss_epoch))
        tStop_epoch = time.time()
        print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
    
    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epoch)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")	


def test(model_path='models/vanilla_state/model-220'):

    batch_size = 2
    print('Start to load data!')
    dataset, img_feature, test_data = get_test_data()
    num_test = test_data['question'].shape[0]
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())+1

    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
            dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0,
            bias_init_vector = None)

    tf_generated_ans, tf_image, tf_question, tf_question_mask, tf_question_length = model.build_generator()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    tf.initialize_all_variables().run()

    num_Y = 0
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
        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
	current_ques_id = test_data['ques_id'][current_batch_file_idx]
        current_img = np.zeros((batch_size, dim_image))
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)

        current_question_mask = np.zeros([max_words_q, batch_size, 2*dim_hidden])
        current_question_mask[current_length_q, range(batch_size), :] = 1 #(max_words_q, batch_size, 2*dim_hidden)

	# do the testing process!!!
        generated_ans = sess.run(
                tf_generated_ans,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question,
                    tf_question_length: current_length_q,
                    tf_question_mask: current_question_mask
                    })
	for i in xrange(0,batch_size):
            ans = dataset['ix_to_ans'][str(generated_ans[i]+1)]
            if(current_ques_id[i] == 0):
                continue
	    # answer: str, question_id: int
            result.append({u'answer': ans, u'question_id': int(current_ques_id[i])})
        tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx)
        print ("Time Cost:", round(tStop - tStart,2), "s")
    
    #accuracy = float(num_Y)/float(num_test);
    print ("Testing done.")
    # Save to JSON
    print ('Saving result...')
    my_list = list(result)
    dd = json.dump(my_list,open(model_name+'.json','w'))
    #print ("Accuracy = ",accuracy)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")

    #return accuracy

if __name__ == '__main__':

    print ("Model name: ",model_name)
    print ("normalize = ",normalize)
    print ("max_words_q = ",max_words_q)
    print ("kernel: ",kernel_w,"*",kernel_h)
    print ("dim_hidden = ",dim_hidden)
    print ("batch_size = ",batch_size)
    print ("learning_rate = ",learning_rate)

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    if TRAIN:
        with tf.device('/gpu:'+gpu_id):
            train()
    if TEST:
        test()

