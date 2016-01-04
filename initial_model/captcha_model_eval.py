"""
Evaluation for captcha.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/captcha_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/will/Desktop/hard_captcha_mean_sub_rrelu',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 3340 + 6511 + 4985 + 1358,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

from captcha_model_train import inputs, inference_captcha_mean_subtracted, MOVING_AVERAGE_DECAY
from utils import get_saver


def eval_once(saver, summary_writer, top_k_ops, summary_op):
    """
    Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_ops: Top K operations for each output.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_counts = [0 for i in range(6)] # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(top_k_ops)
                for i, prediction in enumerate(predictions):
                  true_counts[i] += np.sum(prediction)
                step += 1

            # Compute precision @ 1.
            precisions = [true_count / total_sample_count for true_count in true_counts]
            for i, precision in enumerate(precisions):
                print('%s: %d precision @ 1 = %.3f' % (datetime.now(), i, precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            for i, precision in enumerate(precisions):
                summary.value.add(tag='Precision %d @ 1' % i, simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval food-101 for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels for captchas
        eval_data = FLAGS.eval_data == 'test'
        images, all_labels = inputs(eval_data=eval_data)
        # split the labels
        print(images)
        labels = tf.split(1, 6, all_labels)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _ = inference_captcha_mean_subtracted(images, False, False)
        print(logits[0])
        print(labels[0])
        ls = [tf.reshape(label, [FLAGS.batch_size]) for label in labels]
        print(ls)

        # Calculate predictions.
        top_k_ops = [tf.nn.in_top_k(logit, label, 1) for logit, label in zip(logits, ls)]

        # Restore the moving average version of the learned variables for eval.
        saver = get_saver(MOVING_AVERAGE_DECAY, "bn_")

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                                graph_def=graph_def)

        while True:
            eval_once(saver, summary_writer, top_k_ops, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
