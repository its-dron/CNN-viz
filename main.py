import tensorflow as tf
from tensorflow.contrib import framework
import cv2
import numpy as np

import argparse
from model import build_model, build_train_op

# Resize image to this size for network input
IMAGE_SIZE = [30, 40]

def main(args):
    cam = cv2.VideoCapture(0)

    # Create global step
    global_step = framework.get_or_create_global_step()

    # Tensor that holds raw camera frame
    image_input = tf.placeholder(tf.float32, shape=[None, None, 3], name='input')
    input_size = tf.shape(image_input)[:2]
    batched_input = tf.expand_dims(image_input, 0)
    resized_im = tf.image.resize_images(batched_input, IMAGE_SIZE)

    # Placeholder for target
    label_placeholder = tf.placeholder(tf.float32, shape=[None, None, 3], name='input')
    batched_label = tf.expand_dims(label_placeholder, 0)
    resized_label = tf.image.resize_images(batched_label, IMAGE_SIZE)

    # Create model
    output = build_model(resized_im, is_training=True)
    resized_output = tf.squeeze(tf.image.resize_images(output, input_size))

    # Create optimizer
    train_op, loss_op = build_train_op(resized_label, output, args.learning_rate, global_step=global_step)

    with tf.Session() as sess:
        # Initialize
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        step=0
        while True:
            ret, im = cam.read()
            im = im.astype('float') / 255.0

            import math
            l = math.sin(math.pi * step / 40)
            target = l**2 * im

            step, output, loss, _ = sess.run([global_step, resized_output, loss_op, train_op], feed_dict={image_input: im, label_placeholder: target})

            print "%i: %0.5f" % (step, loss)
            cv2.imshow('input', im)
            cv2.imshow('output', output)
            cv2.imshow('target', target)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Image to be processed. Use webcam if this isn't set."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help='Base learning rate'
    )
    main(parser.parse_args())
