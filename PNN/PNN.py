import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score

class PNN():
    def __init__(self, feature_size, field_size,
                 embedding_size=8,
                 deep_layers=[32, 32], deep_init_size = 50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                greater_is_better=True,
                 use_inner=True):
        assert loss_type in ["logloss","mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_deep = dropout_deep
        self.deep_layer_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [],[]

        self.use_inner = use_inner

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32,shape=[None, None], name='feat_value')

            self.label = tf.placeholder(tf.float32, shape=[None,1],name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # embeddings
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) #N * F * K
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            self.embedding_size = tf.multiply(self.embeddings,feat_value)

            # linear part
            linear_output = []
            for i in range(self.deep_init_size):
                linear_output.append(
                    tf.reshape(
                        tf.reduce_sum(tf.multiply(self.embeddings,self.weights['product-linear'][i]),axis=[1,2]),shape=(-1,1)
                    )
                )
            self.lz = tf.concat(linear_output,axis=1) # N * init_deep_size

            # quardatic signal
            quardatic_output = []
            if self.use_inner:
                for i in range(self.deep_init_size):
                    theta = tf.multiply(self.embeddings,tf.reshape(self.weights['product-quadratic-inner'][i],(1,-1,1))) # N*F*K
                    quardatic_output.append(tf.reshape(tf.norm(tf.reduce_sum(theta,axis=1),axis=1),shape=(-1,1))) # N* 1
            else:
                embedding_sum = tf.reduce_sum(self.embeddings,axis = 1)
                p = tf.matmul(tf.expand_dims(embedding_sum,2), tf.expand_dims(embedding_sum,1)) # N*K*K
                for i in range(self.deep_init_size):
                    theta = tf.multiply(p, tf.expand_dims(self.weights['product-quadratic-outer'][0],0))
                    quardatic_output.append(tf.reshape(tf.reduce_sum(theta,axis=[1,2]),shape=(-1,1))) # N * 1

            self.lp = tf.concat(quardatic_output,axis=1) # N * deep_init_size

            self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz,self.lp),self.weights['product-bias']))
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])

            # deep component
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights['layer_%d' %i]),self.weights['bias_%d' %i])
                self.y_deep = self.deep_layer_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])

            self.out = tf.add(tf.matmul(self.y_deep,self.weights['output']),self.weights['output_bias'])

            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label,self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label,self.out))


            # optimizer
            if self.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            elif self.optimizer == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate= self.learning_rate,momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # nums of params
            total_params = 0
            for var in self.weights.values():
                shape = var.get_shape()
                var_params = 1
                for dim in shape:
                    var_params *= dim.value
                total_params += var_params
            if self.verbose > 0:
                print("#params: %d" % total_params)

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(tf.random_normal(
            [self.feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings'
        )
        weights['feature-bias'] = tf.Variable(tf.random_normal([self.feature_size,1],0.0,0.01),name='feature-_bias')

        # product-layers
        if self.use_inner:
            weights['product-quadratic-inner'] = tf.Variable(
                tf.random_normal([self.deep_init_size,self.field_size],0.0,0.01))
        else:
            weights['product-quadratic-outer'] = tf.Variable(
                tf.random_normal([self.deep_init_size,self.embedding_size,self.embedding_size],0.0,0.01)
            )

        weights['product-linear'] = tf.Variable(tf.random_normal([self.deep_init_size,self.field_size,self.embedding_size],0.0,0.01))
        weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size,],0.0,0.01))
        #deep layers
        num_layers = len(self.deep_layers)
        input_size = self.deep_init_size
        glorots = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorots,size=(input_size,self.deep_layers[0])),dtype=tf.float32
        )
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0,scale=glorots,size=(1,self.deep_layers[0])),dtype=tf.float32)

        for i in range(1,num_layers):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
        weights["layer_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
            dtype=np.float32)  # layers[i-1] * layers[i]
        weights["bias_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
            dtype=np.float32)  # 1 * layer[i]

        return weights

    def get_batch(self,Xi,Xv,y,batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end <= len(y) else len(y)
        return Xi[start:end],Xv[start:end],y[start:end]

    # shuffle three list at the same time
    def shuffle_in_unison_scary(self,a,b,c):
        seed_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(seed_state)
        np.random.shuffle(b)
        np.random.set_state(seed_state)
        np.random.shuffle(c)

    def predict(self, Xi,Xv,y):
        feed_dict={
            self.feat_index:Xi,
            self.feat_value:Xv,
            self.dropout_keep_deep:[1.0] * len(self.dropout_deep),
            self.train_phase:True
        }
        loss = self.sess.run([self.loss],feed_dict=feed_dict)

        return loss

    def fit_on_batch(self,Xi,Xv,y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: self.dropout_dep,
                     self.train_phase: True}

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train,Xv_train,y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch,Xv_batch,y_batch = self.get_batch(Xi_train,Xv_train,y_train,self.batch_size,i)
                self.fit_on_batch(Xi_batch,Xv_batch,y_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape(-1,1)
                loss = self.predict(Xi_valid,Xv_valid,y_valid)
                print("epoch", epoch, "loss", loss)









