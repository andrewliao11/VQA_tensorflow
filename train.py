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
    '''
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
    '''

    return dataset


def train():

    
    dataset = get_data()
    # data['ix_to_word'] is the word pair
    vocabulary_size = len(data['ix_to_word'].keys())



if __name__ == '__main__':
    train()
