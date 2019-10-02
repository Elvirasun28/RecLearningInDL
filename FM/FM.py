from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import tensorflow as tf

class FM:
    def __init__(self,train, test ,k):
        self.x_train, self.ix = FM.vectorize_dic({'users':train.user.values,'items':train.item.values})
        self.X_test, self.ix = FM.vectorize_dic({'users': test.user.values, 'items': test.item.values}, ix, x_train.shape[1])
        ''' Densifing the input matrices'''
        ## you can use tf.SpareseTensor for large sparse datasets
        self.X_train = self.x_train.todense()
        self.X_test = self.X_test.todense()
        self.y_train = train.rating.values
        self.y_test = test.rating.values

        self.n,self.p = self.X_train.shape # n - number of sample, p - features
        self.k = k

    def build_model(self):
        with tf.name_scope('inputs'):
            # design matrix
            X = tf.placeholder('float', shape=[None, self.p])
            # target vector
            y = tf.placeholder('float', shape=[None, 1])
        with tf.name_scope('weights'):
            # bias and weights
            w0 = tf.Variable(tf.zeros([1]))
            W = tf.Variable(tf.zeros([self.p]))

            # interaction factors, randomly initialized
            V = tf.Variable(tf.random_normal([self.k, self.p], stddev=0.01))
        with tf.name_scope('interaction_part'):
            # define how the output values y should be calculated
            linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keepdims=True))
            pair_interactions = (tf.multiply(0.5,
                                             tf.reduce_sum(
                                                 tf.subtract(
                                                     tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                                                     tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                                                 1, keepdims=True)))
            y_hat = tf.add(linear_terms, pair_interactions)

        with tf.name_scope('loss'):
            # l2 regularized sum of squared loss function over w and v
            lambda_w = tf.constant(0.001, name='lambda_w')
            lambda_v = tf.constant(0.001, name='lambda_v')

            l2_norm = (tf.reduce_sum(
                tf.add(
                    tf.multiply(lambda_w, tf.pow(W, 2)),
                    tf.multiply(lambda_v, tf.pow(V, 2))
                )
            ))

            error = tf.reduce_mean(tf.square(y - y_hat))
            loss = tf.add(error, l2_norm)

        with tf.name_scope('op'):
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    def batch(self,):

    @staticmethod
    def vectorize_dic(dic, ix=None, p=None):
        '''
        creates a scipy csr matrix from a list of list (each inner list is
        a set of values corresponding a feature)
        :param dic:dictionary of feature lists. Keys are the name of features
        :param ix:index generator (default None)
        :param p:dimension of featrure space (number of columns in the sparse
        matrix) (default None)
        :return:
        '''

        if ix == None:
            d = count(0)
            ix = defaultdict(lambda: next(d))

        n = len(list(dic.values())[0])  # num samples
        g = len(list(dic.keys()))  # num groups
        nz = n * g  # number of non-zeros

        col_ix = np.empty(nz, dtype=int)

        i = 0
        for k, lis in dic.items():
            ## append index el with k in order to prevent mapping different columns with
            ## same id to same idex
            col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
            i += 1

        row_ix = np.repeat(np.arange(0, n), g)
        data = np.ones(nz)

        if p == None:
            p = len(ix)

        ixx = np.where(col_ix < p)

        return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix



		