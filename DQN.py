import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import os


import Perception
import Reward











class Qnetwork():
    def __init__(self, frame_height, frame_width, frame_channels, num_stacked_frames, total_num_actions):
        flattened_frame_size = frame_height*frame_width*frame_channels*num_stacked_frames
        self.flattened_image = tf.placeholder(shape=[None, flattened_frame_size], dtype=tf.float32)
        #[batch, in_height, in_width, in_channels]
        #[filter_height, filter_width, in_channels, out_channels]
        # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].

        self.state_feature_vector, self.CNN_params = Perception.CNN(input=self.flattened_image,height=frame_height,width=frame_width,in_channel=frame_channels*num_stacked_frames,out_channel=128,weights=[])

        #NOTE :::: Split is not really required, also even if you use split, it should be done on the dimension of feature maps. Also the weight matrices have to be correctly shaped.
        with tf.variable_scope("advantage_stream"):
            self.streamAC = self.state_feature_vector
            self.streamA = tf.contrib.layers.flatten(self.streamAC)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8739)
            self.AW1 = tf.Variable(xavier_init([self.streamA.shape[1].value, 512]), name='FC_1_Weights')
            self.ABias1 = tf.Variable(tf.constant(0.1, shape=[512]), name='FC_1_Bias')
            self.Advantage_FC1 = tf.matmul(self.streamA, self.AW1) + self.ABias1
            self.Advantage_FC1 = tf.nn.relu(self.Advantage_FC1)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8536)
            self.AW2 = tf.Variable(xavier_init([self.Advantage_FC1.shape[1].value, total_num_actions]), name='FC_2_Weights')
            self.ABias2 = tf.Variable(tf.constant(0.1, shape=[total_num_actions]), name='FC_2_Bias')
            self.Advantage = tf.matmul(self.Advantage_FC1, self.AW2) + self.ABias2
            self.advantage_w = [self.AW1, self.ABias1,self.AW2,self.ABias2]
        with tf.variable_scope("value_stream"):
            self.streamVC = self.state_feature_vector
            self.streamV = tf.contrib.layers.flatten(self.streamVC)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8635)
            self.VW1 = tf.Variable(xavier_init([self.streamV.shape[1].value, 512]), name='FC_1_Weights')
            self.VBias1 = tf.Variable(tf.constant(0.1, shape=[512]), name='FC_1_Bias')
            self.Value_FC1 = tf.matmul(self.streamV, self.VW1) + self.VBias1
            self.Value_FC1 = tf.nn.relu(self.Value_FC1)
            xavier_init = tf.contrib.layers.xavier_initializer(seed=8267)
            self.VW2 = tf.Variable(xavier_init([self.Value_FC1.shape[1].value, 1]), name='FC_2_Weights')
            self.VBias2 = tf.Variable(tf.constant(0.1, shape=[1]), name='FC_2_Bias')
            self.Value = tf.matmul(self.Value_FC1, self.VW2) + self.VBias2
            self.value_w = [self.VW1,self.VBias1,self.VW2,self.VBias2]
        # NOTE ::: Add the state value and advantage value to get the q values but note that we subtract the average advantage value from advantage value of all actions.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # NOTE :: we take argmax over advantage values instead of Q values
        self.predict = tf.argmax(self.Advantage, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        network_scope_name = tf.get_variable_scope().name
        if(network_scope_name=='Q_main'):
            Q_vars = self.advantage_w + self.value_w
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, name='main_Q_network_adam_opt')
            self.gvs_Q = self.optimizer.compute_gradients(self.loss,var_list=Q_vars)
            with tf.variable_scope("gradient_clipping_Q_vars"):
                self.capped_gvs_Q = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in self.gvs_Q]
            self.train_Q_op = self.optimizer.apply_gradients(self.capped_gvs_Q, name='Q_vars_grad_update')
            self.gvs_cnn = self.optimizer.compute_gradients(self.loss,var_list=self.CNN_params)
            with tf.variable_scope("gradient_clipping_cnn_vars"):
                self.capped_gvs_cnn = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in self.gvs_cnn]
            self.train_cnn_op = self.optimizer.apply_gradients(self.capped_gvs_cnn, name='CNN_vars_grad_update')
            # self.train_op = self.optimizer.minimize(self.loss)



def target_network_update_op(Q_main_variables, Q_target_variables, tau):
    target_network_update_ops = []
    with tf.variable_scope("target_network_update_ops"):
        for main_network_var, target_network_var in zip(sorted(Q_main_variables, key=lambda v: v.name), sorted(Q_target_variables, key=lambda v: v.name)):
            #print('main var name {0}'.format(main_network_var.name))
            #print('tgt var name {0}'.format(target_network_var))
            assign_value = (main_network_var.value()*tau) + ((1 - tau)*target_network_var.value())
            update_op = target_network_var.assign(assign_value)
            target_network_update_ops.append(update_op)
        grouped_target_network_update_op = tf.group(*target_network_update_ops)
    return grouped_target_network_update_op




