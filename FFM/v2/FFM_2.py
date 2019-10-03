import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle


class FFM(object):
    def __init__(self,args):
        self.k = args.k
        self.f = args.f
        self.p = args.p
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.l2_reg_rate = args.l2_reg_rate
        self.feature2field = args.feature_2field
        self.MODEL_SAVE_PATH = args.MODEL_SAVE_PATH
        self.MODEL_NAME = args.MODEL_NAME

    def build_model(self):
        with tf.variable_scope('inputs',reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder('float32',[self.batch_size,self.p])
            self.y = tf.placeholder('float32',[None,1])

        # linear part
        with tf.variable_scope('linear_layer',reuse=tf.AUTO_REUSE):
            b = tf.get_variable('bias', shape=[1],
                                initializer=tf.zeros_initializer())
            self.w1 = tf.get_variable('w1', shape=[self.p, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.linear_terms = tf.add(tf.matmul(self.X, self.w1), b)
            print('self.linear_terms:')
            print(self.linear_terms)

        # no-linear part
        with tf.variable_scope('nolinear_layer',reuse=tf.AUTO_REUSE):
            self.v = tf.get_variable('v', shape=[self.p, self.f, self.k], dtype='float32',
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # v:pxfxk
            self.field_cross_interaction = tf.constant(0, dtype='float32')
            # 每个特征
            for i in range(self.p):
                # 寻找没有match过的特征，也就是论文中的j = i+1开始
                for j in range(i + 1, self.p):
                    # vifj
                    vifj = self.v[i, self.feature2field[j]]
                    # vjfi
                    vjfi = self.v[j, self.feature2field[i]]
                    # vi · vj
                    vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
                    # xi · xj
                    xixj = tf.multiply(self.X[:, i], self.X[:, j])
                    self.field_cross_interaction += tf.multiply(vivj, xixj)
            self.field_cross_interaction = tf.reshape(self.field_cross_interaction, (self.batch_size, 1))
            print('self.field_cross_interaction:')
            print(self.field_cross_interaction)

        with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
            self.y_out = tf.add(self.linear_terms, self.field_cross_interaction)

        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.log(1 + tf.exp(-self.y * self.y_out)))
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.w1)
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.v)

        with tf.variable_scope('ops',reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, trainable=False)
            opt = tf.train.GradientDescentOptimizer(self.lr)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, x, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.X: x,
            self.y: label
        })
        return loss, step

    def cal(self, sess, x, label):
        y_out_prob_ = sess.run([self.y_out], feed_dict={
            self.X: x,
            self.y: label
        })
        return y_out_prob_, label

    def predict(self, sess, x):
        result = sess.run([self.y_out], feed_dict={
            self.X: x
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)