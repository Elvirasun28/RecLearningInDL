from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from IPython.display import display,Math,Latex
from tqdm import tqdm_notebook as tqdm
#from __future__ import print_function


def vectorize_dic(dic,ix= None, p = None):
    '''
    creates a scipy csr matrix from a list of list (each inner list is
    a set of values corresponding a feature)
    :param dic:dictionary of feature lists. Keys are the name of features
    :param ix:index generator (default None)
    :param p:dimension of featrure space (number of columns in the sparse
    matrix) (default None)
    :return:
    '''

    if ix== None:
        d = count(0)
        ix = defaultdict(lambda :next(d))

    n = len(list(dic.values())[0]) # num samples
    g = len(list(dic.keys())) # num groups
    nz = n * g # number of non-zeros

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        ## append index el with k in order to prevent mapping different columns with
        ## same id to same idex
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(0,n),g)
    data = np.ones(nz)

    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])), shape=(n,p)), ix


''' load dataset '''
# load data with pandas
cols = ['user','item','rating','timestamp']
train = pd.read_csv('FM/data/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('FM/data/ua.test',delimiter = '\t', names=cols)

# vectorize data and convert them to csr matrix
x_train, ix = vectorize_dic({'users':train.user.values,'items':train.item.values})
X_test, ix = vectorize_dic({'users': test.user.values, 'items': test.item.values}, ix, x_train.shape[1])
y_train = train.rating.values
y_test = test.rating.values

''' Densifing the input matrices'''
## you can use tf.SpareseTensor for large sparse datasets
X_train = x_train.todense()
X_test = X_test.todense()

# print(X_train.shape)
# print(X_test.shape)

''' Define FM model with tensorflow '''
n,p = X_train.shape

# number of latent factors
k = 10

# design matrix
X = tf.placeholder('float',shape=[None,p])
# target vector
y = tf.placeholder('float',shape=[None,1])

# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

# interaction factors, randomly initialized
V = tf.Variable(tf.random_normal([k,p],stddev=0.01))

# estimate of y, initialized to 0
y_hat = tf.Variable(tf.zeros([n,1]))

# define how the output values y should be calculated 
linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(W,X),1,keepdims=True))
pair_interactions = (tf.multiply(0.5,
                    tf.reduce_sum(
                        tf.subtract(
                            tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                        1, keepdims=True)))
y_hat = tf.add(linear_terms, pair_interactions)

'''loss function'''
# l2 regularized sum of squared loss function over w and v
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

l2_norm = (tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w,tf.pow(W,2)),
        tf.multiply(lambda_v,tf.pow(V,2))
    )
))

error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error, l2_norm)

''' Optimization '''
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

''' Preparing Batches '''
def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

''' lanching tensorflow graph and training the model '''
epochs = 10
batch_size = 1000

# launch graph
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    # iterate over batches
    for bX, bY in batcher(X_train[perm],y_train[perm],batch_size):
        _,t = sess.run([train_op,loss],feed_dict={X:bX.reshape(-1,p),
                                      y:bY.reshape(-1,1)}
                 )
        print(t)


''' Evaluating the model '''
errors = []
for bX,bY in batcher(X_test, y_test):
    errors.append(sess.run(error,feed_dict={
        X:bX.reshape(-1,p),y:bY.reshape(-1,1)
    }))
RMSE = np.sqrt(np.array(errors)).mean()
print(RMSE)

sess.close()







