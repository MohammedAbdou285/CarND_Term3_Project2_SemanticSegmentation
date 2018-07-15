#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Using tf.saved_model.loader.load to load the vgg model based on its attributes
    tf.saved_model.loader.load(sess=sess, tags=[vgg_tag], export_dir=vgg_path)

    # We need to return the names of the vgg layer, so we get the default graph and use (get_tensor_by_name) function
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # Return a tuple of the loaded layers from vgg16 model
    return input_layer, keep_prob, layer3, layer4, layer7


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Define the L2 Regularization value in order to use it in the following layers
    weights_L2_Regularizer = 1e-3

    '''
    Layer 7 CONV_1x1 and Upsampling
    '''
    # 1x1 Convolution for vgg_16 7th layer in order to reduce the depth to be num_classes
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="SAME",
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_L2_Regularizer),
                                       name="conv_1x1_layer7")

    # Upsampling layer for the output of the previous layer conv_1x1_layer7
    deconv_layer7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, strides=(2, 2), padding="SAME",
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_L2_Regularizer),
                                               name="Deconv_layer7")




    '''
    Layer 4 CONV_1x1, Skip Connection and Upsampling
    '''
    # 1x1 Convolution for vgg_16 4th layer
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="SAME",
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_L2_Regularizer),
                                       name="conv_1x1_layer4")

    # Before the upsampling, we should add the skip connections first
    Skip_Connection_1 = tf.add(conv_1x1_layer4, deconv_layer7, name="Skip_Connection_1")

    # Upsampling layer for the output of the previous layer conv_1x1_layer7
    deconv_layer4 = tf.layers.conv2d_transpose(Skip_Connection_1, num_classes, 4, strides=(2, 2), padding="SAME",
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_L2_Regularizer),
                                               name="Deconv_layer4")

    '''
    Layer 4 CONV_1x1, Skip Connection and Upsampling
    '''
    # 1x1 Convolution for vgg_16 3rd layer
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="SAME",
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_L2_Regularizer),
                                       name="conv_1x1_layer3")

    # Before the upsampling, we should add the skip connections first
    Skip_Connection_2 = tf.add(conv_1x1_layer3, deconv_layer4, name="Skip_Connection_2")

    # Upsampling layer for the output of the previous layer conv_1x1_layer7
    deconv_layer3 = tf.layers.conv2d_transpose(Skip_Connection_2, num_classes, 16, strides=(8, 8), padding="SAME",
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_L2_Regularizer),
                                               name="Deconv_layer3")

    # Return th eoutput of the model
    return deconv_layer3


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape the NN output (logits) and labels to be a 2D instead of having 4D tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Create the Loss Function that will be passed to the Optimizer
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # Define Adam optimizer to be used
    Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam")

    # Apply OPtimizer to the Loss Function
    train_op = Optimizer.minimize(loss=Loss,name="Optimization")

    # Return tuple of the logits, train_op, and cross entropy loss
    return logits, train_op, Loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    # Run the global variables initializers
    sess.run(tf.global_variables_initializer())

    # For each epoch, we will iterate through the batches based on the batch_size
    for epoch in range(epochs):
        print("Epoch (",epoch,"): ")
        loss_log = []
        for image, label in get_batches_fn(batch_size):
            # Prepare the feed_dict that will be used in the sess.run
            feed_dict = {input_image:image, correct_label:label, keep_prob:keep_prob, learning_rate:learning_rate}
            # Run the session based on the previous feed_dict attributes
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            loss_log.append(loss)
        # Printing loss values for each epoch
        print("     loss = ", loss_log)
        # Going to the Next line in the following epoch
        print()
    print("Training Finished...")



tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # We need to create placholders for correct_label and learning rate which are needed in optimize function
        correct_label = tf.placeholder(tf.int32, shape=[None,None,None, num_classes], name="Correct_Label")
        learning_rate = tf.placeholder(tf.float32, shape=None, name="Learning_Rate")

        # Calling the functions that we have created previously
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess=sess,vgg_path=vgg_path)
        final_layer_output = layers(vgg_layer3_out=layer3_out, vgg_layer4_out=layer4_out, vgg_layer7_out=layer7_out, num_classes=num_classes)
        logits, train_op, Loss = optimize(nn_last_layer=final_layer_output,correct_label=correct_label,learning_rate=learning_rate, num_classes=num_classes)

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
