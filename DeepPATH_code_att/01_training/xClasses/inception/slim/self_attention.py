from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def Add_self_att(net, block_scope=None):
    att_block_scope = 'att_block' if block_scope is None else block_scope + '_att'
    net = self_att_block(net, att_block_scope)
    return net
#ratio reduce the para -> efficiency
#input_feature: (N,H,W,C) feature maps
# query, key, att
def self_att_block(input_feature, name, ratio=1):

    #To get Delving Deep into Rectifiers (also know as the "MSRA initialization"), use (Default):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    #bias_initializer = tf.constant_initializer(value=0.0)
    # W1 = c*R1 = C*1
    # W2 = C*R2 = C*C
    '''with statement in Python is used in exception handling to make the code cleaner and much more readable
    '''
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        R1 = 1
        R2 = channel
        # Global average pooling (N,H,W,C)
        chanel_embed = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)

        query = tf.layers.dense(inputs=chanel_embed,
                                     units=R1,
                                     #activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     #bias_initializer=bias_initializer,
                                     name='query')

        '''key = tf.layers.dense(inputs=chanel_embed,
                                     units= R2,
                                     #activation=tf.nn.softmax,
                                     kernel_initializer=kernel_initializer,
                                     #bias_initializer=bias_initializer,
                                     name='key')
        '''
        key = chanel_embed
        att_score=tf.matmul(query, key)

        att_prob= tf.nn.softmax(att_score)
        att_output = input_feature * att_prob
    return att_output