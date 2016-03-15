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
model_path = '/home/andrewliao11/Work/VQA_challenge/model/'


## Some args
normalize = True
max_words_q = 26
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
	# convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
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
	label = tf.placeholder(tf.int32, [self.batch_size]) # (batch_size, )

	# [image] embed image feature to dim_hidden
        image_emb = tf.nn.xw_plus_b(image, self.embed_image_W, self.embed_image_b) # (batch_size, dim_hidden)
        image_emb = tf.nn.dropout(image_emb, self.drop_out_rate)
        image_emb = tf.tanh(image_emb)

	# [answer] ground truth
        labels = tf.expand_dims(label, 1) # (batch_size, 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
        concated = tf.concat(1, [indices, labels]) # (batch_size, 2)
        answer = tf.sparse_to_dense(concated, tf.pack([self.batch_size, num_answer]), 1.0, 0.0) # (batch_size, num_answer)
	
	'''
	# [question_mask] 
	question_length_exp = tf.expand_dims(question_length, 1) # b x 1
        indices_q = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
        concated_tem = tf.concat(1, [indices_q, question_length_exp]) # b x 2
	question_length_exp_exp = tf.expand_dims(concated_tem, 2) # (batch_size, 2, 1)
	indices_2dim_hidden_exp = tf.expand_dims(tf.range(0, 2*dim_hidden, 1), 0) # (1, 2*dim_hidden)
	indices_2dim_hidden_exp_exp = tf.expand_dims(indices_2dim_hidden_exp, 0) # (1, 1, 2*dim_hidden)
	concated_q = tf.add(question_length_exp_exp,indices_2dim_hidden_exp_exp) # (batch_size, 2, 2*dim_hidden)
	#concate = tf.concat(2, [question_length_exp_exp, indices_2dim_hidden_exp_exp]) # (batch_size, 2, 2*dim_hidden)
        question_mask = tf.sparse_to_dense(concated_q, tf.pack([self.batch_size, max_words_q, 2*dim_hidden]), 1.0, 0.0) # (batch_size, max_words_q, 2*dim_hidden)
	
	# [question_mask] 
	labels_q = tf.expand_dims(question_length, 1) # b x 1
        indices_q = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
        concated_q = tf.concat(1, [indices_q, labels_q]) # b x 2
        question_mask = tf.sparse_to_dense(concated_q, tf.pack([self.batch_size, max_words_q]), 1.0, 0.0) # (batch_size, max_words_q)
	#question_mask_tem_tem = tf.expand_dims(question_mask_tem, 2) # (batch_size, max_words_q,  1)
	#question_mask_ = tf.tile(question_mask_tem_tem, 2*dim_hidden) # (batch_size, max_words_q, 2*dim_hidden)
	'''
	probs = []
        loss = 0.0

	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        #state = tf.zeros([self.batch_size, max_words_q]) 
	#output = tf.zeros([self.batch_size, 2*self.dim_hidden])

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
	    #output, state = self.stacked_lstm(image_emb, state)
	    #output, state = self.stacked_lstm(tf.concat(1,[image_emb, question_emb]), state)

	    # record the state an output
	    states.append(state2)
	    outputs.append(output2)
	    	
	
	# predict   
	# pack -> convert input into an array
	output = tf.pack(outputs) # (max_words_q, batch_size, 4*dim_hidden)
	'''
	tem = []
	for i in range(self.batch_size):
	    tem.append(outputs[:,i,:])
	tem = tf.pack(tem)  # (batch_size, max_words_q, 2*dim_hidden)
	'''
	#for i in range(self.batch_size):
	'''
	states = tf.gather(states, question_length)
	print (states)
	states = tf.gather(range(10), states)
	print (states)
	'''
	output_final = tf.reduce_sum(tf.mul(output, tf.to_float(question_mask)), 0) # (batch_size, 2*dim_hidden)
	answer_pred = tf.nn.xw_plus_b(output_final, self.answer_emb_W, self.answer_emb_b) # (batch_size, num_answer)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(answer_pred, answer) # (batch_size, )
	loss = tf.reduce_sum(cross_entropy)
	loss = loss/self.batch_size

	return loss, image, question, question_mask, question_length, label
    '''
    def build_generator(self):
	
        # placeholder is for feeding data
        image = tf.placeholder(tf.float32, [1, self.dim_image])  # (1, dim_image)
        question = tf.placeholder(tf.int32, [1, max_words_q])
        question_length = tf.placeholder(tf.int32, [1])
        question_mask = tf.placeholder(tf.int32, [1, max_words_q, self.dim_image])

        # [image] embed image feature to dim_hidden
        image_emb = tf.nn.xw_plus_b(image, self.embed_image_W, self.embed_image_b) # (1, dim_hidden)
        image_emb = tf.nn.dropout(image_emb, self.drop_out_rate)
        image_emb = tf.tanh(image_emb)

	# [question_mask]
        labels_q = tf.expand_dims(question_length, 1) # (1, 1)
        indices_q = tf.expand_dims(tf.range(0, 1, 1), 1) # (1, 1)
        concated_q = tf.concat(1, [indices_q, labels_q]) # (1, 2)
        question_mask = tf.sparse_to_dense(concated_q, tf.pack([self.batch_size, max_words_q]), 1.0, 0.0) # (1, max_words_q)
        question_mask = tf.expand_dims(question_mask, 2) # (1, max_words_q,  1)
        question_mask = tf.tile(question_mask, 2*dim_hidden) # (1, max_words_q, 2*dim_hidden)

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])

	#generated_answer = []
	outputs = []
	states = []

	for j in range(max_words_q):
	    tf.get_variable_scope().reuse_variables()
            if j == 0:
                question_emb = tf.zeros([1, self.dim_hidden])
            else:
                with tf.device("/cpu:0"):
                    question_emb = tf.nn.embedding_lookup(self.question_emb_W, question[:,j-1]) #(dim_hidden, )
		    question_emb = tf.expand_dims(question_emb, 0) # (1, dim_hidden)
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout(tf.concat(1,[image_emb, question_emb]), state1 )
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout(output1, state2 )

            # record the state an output
            states.append(state2)
            outputs.append(output2)
	
	# predict
        outputs = tf.pack(outputs) # (max_words_q, 1, 2*dim_hidden)
        tem = outputs[:,1,:] # (max_words_q, 2*dim_hidden)
	tem = tf.expands_dims(tem, 0) # (1, max_words_q, 2*dim_hidden)
        output_final = tf.reduce_sum(tf.mul(tem, question_mask), 1) # (1, 2*dim_hidden)
        answer_pred = tf.nn.xw_plus_b(output_final, self.answer_emb_W, self.answer_emb_b) # (1, num_answer)

        max_prob_index = tf.argmax(answers_pred, 1)[0]
        generated_answer = max_prob_index
    
        return generated_answer, image, question, question_length
    '''

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
    
    tf_loss, tf_image, tf_question, tf_question_mask, tf_question_length, tf_label = model.build_model()
    
    sess = tf.InteractiveSession()
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
	train_data['answers'] = train_data['answers'][index]
        train_data['img_list'] = train_data['img_list'][index]
	
        tStart_epoch = time.time()
        loss_epoch = np.zeros(num_train)
	#num_batch = num_train/batch_size + 1
	#split_batch = np.array_split(np.arange(num_train),num_batch)
	#for current_batch_file_idx in split_batch:
	for current_batch_start_idx in xrange(0,num_train-1,batch_size):
            tStart = time.time()
            # set data into current*
	    if current_batch_start_idx + 10 < num_train:
		current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+10)
	    else:
		current_batch_file_idx = range(current_batch_start_idx,num_train)
	    current_question = train_data['question'][current_batch_file_idx,:]
            #current_wuestion_mask = question_mask[:,current_batch_file_idx,:]
            current_length_q = train_data['length_q'][current_batch_file_idx]
            current_answers = train_data['answers'][current_batch_file_idx]
            current_img_list = train_data['img_list'][current_batch_file_idx]
	    current_img = np.zeros((batch_size, dim_image))
	    current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
	   
	    current_question_mask = np.zeros([max_words_q, batch_size, 2*dim_hidden])
    	    current_question_mask[current_length_q, range(batch_size), :] = 1 #(max_words_q, batch_size, 2*dim_hidden)
	    '''
            # one batch at a time
            #for idx_batch in xrange(batch_size):
                #current_img_idx[idx_batch] = current_img_list[idx_batch] # (batch_size, )
		# minus 1 since in MSCOCO the idx is 0~82459
		# 		   VQA 	  the idx is 1~82460	
		#current_img[idx_batch,:] = img_feature[current_img_idx-1] # (batch_size, dim_image)
		
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
			tf_question_mask: current_question_mask,
                        tf_label: current_answers
                        })
	    loss_epoch[current_batch_file_idx] = loss
            tStop = time.time()
            print ("Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")

	'''
	# every 20 epoch: print result
        if np.mod(epoch, 20) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

            current_batch = h5py.File(test_data[np.random.randint(0,len(test_data))])
            video_tf, video_mask_tf, HLness_tf, caption_tf, probs_tf, last_embed_tf, state1_tf, state2_tf = model.build_generator()
            ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())
            mp = []
            pred_sent = []
            gt_sent = []
            gt_captions = current_batch['title']
	'''
'''
def test(model_path='models/model-900', video_feat_path=video_feat_path):


    print('Start to load data!')
    dataset, img_feature, train_data = get_data()
    num_test = train_data['question'].shape[0]
    # count question and caption vocabulary size
    vocabulary_size_q = len(dataset['ix_to_word'].keys())

    model = Answer_Generator(
            dim_image = dim_image,
            n_words_q = vocabulary_size_q,
            dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0,
            bias_init_vector = None)
	
    generated_answer_tf, image_tf, question_tf, question_length_tf = model.build_generator()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # TODO
    mp = []
    pred_sent = []
    gt_sent = []
    HLness = []
    for idx, video_feat_path in enumerate(test_data):
        print video_feat_path
        test_data_batch = h5py.File(video_feat_path)
        gt_captions = test_data_batch['title']
        for xxx in xrange(test_data_batch['label'].shape[1]):
                video_feat = np.zeros((1, n_frame_step, dim_image))
                video_mask = np.zeros((1, n_frame_step))
                video_feat[0,:,:] = test_data_batch['data'][:,xxx,:]
                idx = np.where(test_data_batch['label'][:,xxx] != -1)[0]
                video_mask[0,idx] = 1

                #generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                generated_HL_index = sess.run(HLness_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                state1, state1_tmp = sess.run([state1_tf,state2_tf], feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                #print state1[0,:100]
                #print generated_word_index
                #HL_words = sess.run(HL_words_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                #generated_HL_index = HL_words[0]
                #generated_word_index = np.array(generated_HL_index)[n_frame_step:]
                #print HL_words
                #print generated_word_index, generated_HL_index
                probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
                embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
'''



if __name__ == '__main__':
    #with tf.device('/gpu:'+str(6)):
    #    train()
    train()
