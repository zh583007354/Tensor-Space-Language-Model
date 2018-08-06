#encoding =utf-8
import tensorflow as tf 
 

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "data",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: BASIC and Tensor"
                    )
FLAGS = flags.FLAGS

BASIC = "basic"
TENSOR = "Tensor"


class Config(object):
  
  init_scale = 0.1
  learning_rate = 0.002
  max_grad_norm = 10
  num_layers = 1
  num_steps = 20
  hidden_size = 16
  max_epoch = 5
  max_max_epoch = 1000
  keep_prob = 0.5
  lr_decay = 0.9
  batch_size = 20
  # vocab_size = 267735
  # vocab_size = 33230
  # vocab_size = 20000
  vocab_size = 10000
  rnn_mode = BASIC  

class TensorConfig(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = TENSOR


