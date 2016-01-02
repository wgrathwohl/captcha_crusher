"""
This file contains classes and methods that allow us to run
trained models in production
"""
import tensorflow as tf
import utils
import os
import numpy as np
from captcha_model_train import inference_captcha, inference_captcha_mean_subtracted, MOVING_AVERAGE_DECAY
import Image
import sys
import json


def live_inference_captcha_hard(image):
    reshaped_image = tf.cast(image, tf.float32)
    height = 60
    width = 190
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image, height, width
    )
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    float_image_batch = tf.reshape(float_image, (1, height, width, 1))
    logits, _ = inference_captcha_mean_subtracted(float_image_batch, False, False, num_classes=36)
    return logits


def live_inference_captcha_easy(image):
    reshaped_image = tf.cast(image, tf.float32)
    height = 60
    width = 190
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image, height, width
    )
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    float_image_batch = tf.reshape(float_image, (1, height, width, 1))
    logits, _ = inference_captcha(float_image_batch, False, False, num_classes=26)
    return logits


class LiveModel:
    def __init__(self, checkpoint_path, logits, im_shape, num_channels, ind_to_label, moving_average_decay, batch_size=1):
        """
        checkpoint_path: path to model checkpoint file
        logits is a function taking a tensor -> float array
        logits must take a single tensor argument

        ind_to_label is object such that ind_to_label[ind] = label
        """
        self.im_shape = im_shape
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.Session()
            self.checkpoint_step = checkpoint_path.split('-')[-1]
            self.checkpoint_path = checkpoint_path
            self.images_placeholder = tf.placeholder("float", shape=[im_shape[0], im_shape[1], num_channels])
            self.logits = logits(self.images_placeholder)
            self.saver = utils.get_saver(moving_average_decay, nontrainable_restore_names="bn_")
            self.ind_to_label = ind_to_label

        # restore the model's parameters
        self.saver.restore(self.sess, checkpoint_path)

    def get_image_logits(self, image):
        """
        Runs the image through the network and returns the logits
        """
        fd = {self.images_placeholder: image}
        image_logits = self.sess.run(self.logits, feed_dict=fd)
        return image_logits

    def get_image_labels(self, image):
        """
        returns a list of labels for the image
        """
        logits = self.get_image_logits(image)
        softmaxs = []
        for logit in logits:
            num = np.exp(logit[0])
            den = sum(num)
            softmax = num / den
            softmaxs.append(softmax)
        inds = [np.argmax(logit) for logit in logits]
        pred = [self.ind_to_label[ind] for ind in inds]
        confs = [softmax[ind] for softmax, ind in zip(softmaxs, inds)]
        c = 1
        for conf in confs:
            c *= conf
        return pred, c


# the ind -> label maps
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTERS_PLUS_NUMBERS = 'abcdefghijklmnopqrstuvwxyz0123456789'


if __name__ == "__main__":
    # print __file__, "FILE"
    # abspath = os.path.abspath(__file__)
    # dir_path = abspath.split('initial_model/live_model.py')[0]
    # model_path = os.path.join(dir_path, "models")
    # load the live models
    with tf.variable_scope("hard_model"):
        lm_h = LiveModel(
            os.path.join('/home/will/Desktop/hard_captcha_mean_sub/model.ckpt-7600'),
            live_inference_captcha_hard, (70, 200), 1, LETTERS_PLUS_NUMBERS, MOVING_AVERAGE_DECAY
        )
    dname = "/home/will/Desktop/hard_captchas_test/"
    ims = os.listdir(dname)
    c = 0
    import shutil
    import matplotlib.pyplot as plt
    for i, im in enumerate(ims):
        #print im
        imp = os.path.join(dname, im)
        image = Image.open(imp)
        actual = im.split('-')[-1].split('.')[0]
        im = np.asarray(image)
        im = np.reshape(im[:, :, 0], [70, 200, 1])
        pred, prob = lm_h.get_image_labels(im)
        #print pred
        pred = str.join('', pred)
        #print actual, pred, actual == pred
        #print ''
        if actual.lower() != pred:
            c += 1
            #shutil.copy(imp, "/tmp/bad")
            print actual, pred, actual == pred
            #plt.imshow(image)
            #print(pred)
            #plt.show()
        #else:
            #shutil.copy(imp, "/tmp/good")
    print("GOT ACCURACY: {}".format(1 - float(c) / len(ims)))


    # with tf.variable_scope("easy_model"):
    #     lm_e = LiveModel(
    #         os.path.join(model_path, 'easy_model.ckpt-8000'),
    #         live_inference_captcha_easy, (70, 200), 1, LETTERS, MOVING_AVERAGE_DECAY
    #     )

    # if len(sys.argv) != 3:
    #     print "Usage: 'python run_captcha.py filenames.txt output.txt' where filenames.txt contains names of image files that you want labels for"

    # fname = sys.argv[1]
    # out_fname = sys.argv[2]
    # res = []

    # with open(fname, 'r') as f:
    #     for line in f:
    #         line = line.strip()
    #         im = Image.open(line)
    #         np_im = np.asarray(im)
    #         if len(np_im.shape) == 3:
    #             # this is a hard captcha
    #             im = np.reshape(np_im[:, :, 0], [70, 200, 1])
    #             pred, conf = lm_h.get_image_labels(im)
    #             pred = str.join('', pred)
    #             j = {}
    #             j['filename'] = line
    #             j['labels'] = pred
    #             j['confidence'] = conf
    #             j['type'] = "lowercase-letters+numbers"
    #             res.append(j)
    #         else:
    #             # this is an easy captcha
    #             im = np.reshape(np_im, [70, 200, 1])
    #             pred, conf = lm_e.get_image_labels(im)
    #             pred = str.join('', pred)
    #             j = {}
    #             j['filename'] = line
    #             j['labels'] = pred
    #             j['confidence'] = conf
    #             j['type'] = "capital-letters"
    #             res.append(j)
    # with open(out_fname, 'w') as f:
    #     for j in res:
    #         f.write(json.dumps(j) + '\n')
