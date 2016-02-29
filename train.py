from __future__ import print_function
import h5py
import numpy as np
import json

json_data_path= '/home/andrewliao11/VQA_LSTM_CNN/data_prepro.json'
h5_data_path = '/home/andrewliao11/VQA_LSTM_CNN/data_prepro.h5'
image_feature_path = '/home/andrewliao11/VQA_LSTM_CNN/data_img.h5'

## Some args
normalize = True
def get_data():

    dataset = {}

    # load json file
    with open(json_data_path) as data_file:
	data = json.load(data_file)
    for key in data.keys():
	dataset[key] = data[key]
   
    # load image feature
    with h5py.File(image_feature_path,'r') as hf:
        tem = hf.get('images_train')
        dataset['fv_im'] = np.array(tem)
    # load h5 file
    with h5py.File(h5_data_path,'r') as hf:
	tem = hf.get('ques_train')
	dataset['question'] = np.array(tem)
	tem = hf.get('ques_length_train')
        dataset['lengths_q'] = np.array(tem)
	tem = hf.get('img_pos_train')
        dataset['img_list'] = np.array(tem)
	tem = hf.get('answers')
        dataset['answers'] = np.array(tem)
    
    print(dataset['fv_im'].shape)
    #print(np.multiply(dataset['fv_im'],dataset['fv_im']).shape)
    if normalize:
	tem =  np.sqrt(np.sum(np.multiply(dataset['fv_im'],dataset['fv_im'])))
	print(tem)
	dataset['fv_im'] = np.divide(dataset['fv_im'],np.tile(tem,(1,4096)))
	#local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2))
        #dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,4096)):float()

    return dataset

class Answer_generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

	# LSTM cell
	self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden,self.dim_hidden,use_peepholes = True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden,2*self.dim_hidden,use_peepholes = True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)

	# word embedded
	self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
    def build_model(self):
	# placeholder is for feeding data
	image_feature = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image])
	# TODO
	caption_emb = tf.placeholder(tf.int32, [self.batch_size, ??])       
	question_emb = tf.placeholder(tf.int32, [self.batch_size, ??])

	# LSTM state 
	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        state1_return = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2_return = tf.zeros([self.batch_size, self.lstm2.state_size])

    def build_generator(self):
	# placeholder si for feeding data
	image_feature = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
	# TODO
        caption_emb = tf.placeholder(tf.int32, [1, ??])
        question_emb = tf.placeholder(tf.int32, [1, ??])

	# LSTM state
        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        state1_return = tf.zeros([1, self.lstm1.state_size])
        state2_return = tf.zeros([1, self.lstm2.state_size])

def train():

    
    dataset = get_data()
    # data['ix_to_word'] is the word pair
    vocabulary_size = len(data['ix_to_word'].keys())



if __name__ == '__main__':
    train()
