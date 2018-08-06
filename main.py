# encoding = utf-8
import time
import numpy as np
import tensorflow as tf
import reader
import util
from tensorflow.python.client import device_lib
from RNNmodel import RNNModel
from config import *
class PTBInput(object):

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, s in enumerate(model.initial_state):
      feed_dict[s] = state[i]
      # feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps
    #iters += 1
    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs/iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  """Get model config."""
  config = Config()
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
  raw_data = reader.ptb_raw_data(FLAGS.data_path)

  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = RNNModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = RNNModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = RNNModel(is_training=False, config=config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    config_proto.gpu_options.allow_growth = True
    with sv.managed_session(config=config_proto) as session:
      best_valid_perplexity = 10000
      valid_perplexity = 0
      best_test_perplexity = 10000
      test_perplexity = 0
      for i in range(config.max_max_epoch):
        if valid_perplexity > best_valid_perplexity or test_perplexity > best_test_perplexity:
          # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          if config.learning_rate > 0.0001:
            config.learning_rate = config.learning_rate * config.lr_decay
          else:
            config.learning_rate = config.learning_rate
        else:
          config.learning_rate = config.learning_rate
        m.assign_lr(session, config.learning_rate)
        print("Epoch: %d Learning rate: %.4f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        if valid_perplexity < best_valid_perplexity:
          best_valid_perplexity = valid_perplexity
        print("Epoch: %d Valid Perplexity: %.3f best valid: %.3f" % (i + 1, valid_perplexity, best_valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        if test_perplexity < best_test_perplexity:
          best_test_perplexity = test_perplexity
          f = open('ppl_hidden_'+str(config.hidden_size)+'.txt', 'w')
          f.write('best_test_perplexity:'+str(best_test_perplexity)+'\n')
          f.write('best_valid_perplexity:'+str(best_valid_perplexity)+'\n')
          f.close()
        print("Epoch: %d Test Perplexity: %.3f best test: %.3f" % (i + 1, test_perplexity, best_test_perplexity))
       
        
        

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
