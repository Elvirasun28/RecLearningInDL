from FFM.v2.FFM_2 import FFM,Args
from FFM.v2.util import get_batch,transfer_data
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf


## loading base params
args = Args()

## load dataset
train_data_path = 'FFM/data/train_sample.csv'
train_data = pd.read_csv(train_data_path)
train_data['click'] = train_data['click'].map(lambda x: -1 if x == 0 else x) ## because we use logistic loss for optimizating
# loading feature2field dict
with open('FFM/data/feature2field.pkl','rb') as f:
    args.feature_2field = pickle.load(f)
f.close()

fields = ['C1', 'C18', 'C16', 'click']
fields_dict = {}
for field in fields:
    with open('FFM/data/' + field + '.pkl', 'rb') as f:
        fields_dict[field] = pickle.load(f)

args.f = len(fields) - 1
args.p = max(fields_dict['click'].values()) - 1

all_len = max(fields_dict['click'].values()) + 1

cnt = train_data.shape[0] // args.batch_size

with tf.Session() as sess:
    model = FFM(args)
    model.build_model()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if args.is_training:
        # batch_size data
        for i in range(args.epoch):
            for j in range(cnt):
                data = get_batch(train_data,args.batch_size,j)
                actual_batch_size = len(data)
                batch_X = []
                batch_y = []
                for k in range(actual_batch_size):
                    sample = data.iloc[k,:]
                    array = transfer_data(sample, fields_dict,all_len)
                    batch_X.append(array[:-2])
                    batch_y.append(array[-1])
                batch_X = np.array(batch_X)
                batch_y = np.array(batch_y)
                batch_y = batch_y.reshape(args.batch_size,1)
                loss,step = model.train(sess,batch_X,batch_y)
                if j % 100 == 0:

