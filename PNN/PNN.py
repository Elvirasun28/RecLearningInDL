import numpy as np
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

