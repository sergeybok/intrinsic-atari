import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle
import time
import datetime
import sys
import cv2
from scipy import misc 

import Perception
import Reward
import DQN 

import gym 






use_intrinsic_reward = False
use_complete_random_agent = True
historical_sample_size = 100


gym_environment_name = 'Breakout-v0'

# ### Training the network
# Setting all the training parameters
batch_size = 100  # How many experiences to use for each training step.
gamma_discount_factor = .9999  # Discount factor on the target Q-values
startE = 0.5  # Starting chance of random action
endE = 0.05  # Final chance of random action
annealing_steps = 7000  # How many steps of training to reduce startE to endE.
batch_size_deconv_compressor = 4

intrinsic_reward_rescaling_factor = 10
num_episodes = 1300  # How many episodes of game environment to train network with.
if(use_complete_random_agent):
    update_freq_per_episodes = num_episodes # How often to perform a training step.
else:
    update_freq_per_episodes = 25  # How often to perform a training step.
pre_train_steps = 100  # How many steps of random actions before training begins.
max_actions_per_episode = 160  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path_Complete_Network = "./curiosity_model/intinsic_model"  # The path to save our model to.
# path_Frame_Predictor = "./curiosity_model/frame_predictor_model"  # The path to save our model to.
model_saving_freq = 200
# h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network
num_stacked_frames = 4
previous_frames = []

state_frame_normalization_factor = 255.0



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# ### Experience Replay
# This class allows us to store experiences and sample then randomly to train the network.
class experience_buffer():
    def __init__(self, buffer_size=2000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        #Since we have 5 elements in the experience : s, a, r, s1, is_terminal_flag, therefore we reshape it as [size, 5]
        if(size>len(self.buffer)):
            size = len(self.buffer)
        return (np.reshape(np.array(random.sample(self.buffer, size)), [size, 5]), size)

    def is_empty(self):
        return len(self.buffer) == 0


print('building environment {0}..'.format(gym_environment_name))

env = gym.make(gym_environment_name)

frame_height = env.observation_space.shape[0]
frame_width = env.observation_space.shape[1]

frame_channels = 1 # Need to hardcode grayscale

total_num_actions = env.action_space.n 


print('building model..')



with tf.variable_scope("Q_main") as Q_main_scope:
    mainQN = DQN.Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, 
    					num_stacked_frames=num_stacked_frames, total_num_actions=total_num_actions)
with tf.variable_scope("Q_target") as Q_target_scope:
    targetQN = DQN.Qnetwork(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, 
    					num_stacked_frames=num_stacked_frames, total_num_actions=total_num_actions)



saver = tf.train.Saver()

#IMPORTANT NOTE::: When a scope is passed to the following tf.get_collection it returns only those trainable variables which are named.
# So it is important to name all the trainable variables.
Q_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Q_main_scope.name)
Q_target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Q_target_scope.name)
grouped_target_network_update_op = DQN.target_network_update_op(Q_main_variables, Q_target_variables, tau)


if use_intrinsic_reward:
    curiosity = Reward.Compressor(frame_height=frame_height, frame_width=frame_width, frame_channels=frame_channels, 
    						num_stacked_frames = num_stacked_frames,
                            state_feature_size=mainQN.state_feature_vector.shape[1].value, CNN_W=mainQN.CNN_params,
                            total_num_actions=total_num_actions, network_name='compressor')


myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = float(startE - endE)/annealing_steps

# create lists to contain total rewards and steps per episode
steps_taken_per_episode_list = []
reward_per_episode_list = []
mean_reward_window_len = 10
mean_reward_per_episode_window_list = []
total_steps = 0


# Make a path for our model to be saved in.
if not os.path.exists(path_Complete_Network):
    os.makedirs(path_Complete_Network)

with tf.variable_scope("reward_scalars") as reward_scalars:
    curr_episode_total_reward_placeholder = tf.placeholder(tf.float32, name='curr_episode_total_reward')
    curr_episode_reward_summary = tf.summary.scalar("Per episode reward", curr_episode_total_reward_placeholder)
    mean_reward_over_window_placeholder = tf.placeholder(tf.float32, name='mean_reward_over_window')
    mean_reward_over_window_summary = tf.summary.scalar("Mean episodic reward over window", mean_reward_over_window_placeholder)
    avg_batch_intrinsic_reward_placeholder = tf.placeholder(tf.float32, name='avg_batch_intrinsic_reward')
    avg_batch_intrinsic_reward_summary = tf.summary.scalar("Avg batch intrinsic reward", avg_batch_intrinsic_reward_placeholder)

with tf.variable_scope("loss_scalars") as loss_scalars:
    avg_batch_compressor_loss_placeholder = tf.placeholder(tf.float32, name='avg_batch_compressor_loss')
    avg_batch_compressor_loss_summary = tf.summary.scalar("Avg batch compressor loss", avg_batch_compressor_loss_placeholder)
    avg_batch_DQN_loss_placeholder = tf.placeholder(tf.float32, name='avg_batch_DQN_loss')
    avg_batch_DQN_loss_summary = tf.summary.scalar("Avg batch DQN loss", avg_batch_DQN_loss_placeholder)

"""
with tf.variable_scope("images") as image_summaries:
    max_images_to_display = 5
    with tf.variable_scope("input_images") as input_images_summaries:
        input_frame_summary = tf.summary.image("input_frame", tf.reshape(mainQN.flattened_image, shape=[-1, frame_height, frame_width, frame_channels]), max_images_to_display)
    with tf.variable_scope("cnn_features") as cnn_features_summaries:
        max_images_to_display = 1
        conv_layer_summary_list = []
        for idx, conv_layer in enumerate([mainQN.conv1, mainQN.conv2, mainQN.conv3]):
            with tf.variable_scope("conv_layer_"+str(idx+1)) as conv_layer_summaries:
                conv_feature_summary_list = []
                num_features = conv_layer.shape[3].value
                for i in range(num_features):
                    conv_feature_summary = tf.summary.image("cnn_"+str(idx+1)+"_feature_"+str(i), tf.slice(conv_layer, [0, 0, 0, i], [-1, conv_layer.shape[1].value, conv_layer.shape[2].value, 1]), max_images_to_display)
                    conv_feature_summary_list.append(conv_feature_summary)
                conv_feature_merged = tf.summary.merge(conv_feature_summary_list)
                conv_layer_summary_list.append(conv_feature_merged)
        cnn_merged_summaries = tf.summary.merge(conv_layer_summary_list)
"""

with tf.name_scope("Q_network_weights_summary") as Q_network_weights_summaries:
    with tf.name_scope("FC_weights") as mainQN_fully_connected_layers_weights_summaries:
        mainQN_FC_weights_list = [mainQN.AW1, mainQN.ABias1, mainQN.AW2, mainQN.ABias2, mainQN.VW1, mainQN.VBias1, mainQN.VW2, mainQN.VBias2]
        mainQN_FC_weights_summary_list = []
        for var in mainQN_FC_weights_list:
            mainQN_FC_weights_summary_list.append(tf.summary.histogram(var.name, var))
        merged_FC_weights_mainQN_summary = tf.summary.merge(mainQN_FC_weights_summary_list)
    with tf.name_scope("CNN_weights") as cnn_weights_summaries:
        cnn_weights_summary_list = []
        for var in mainQN.CNN_params:
            cnn_weights_summary_list.append(tf.summary.histogram(var.name, var))
        merged_cnn_weights_summary = tf.summary.merge(cnn_weights_summary_list)
    merged_weights_mainQN_summary = tf.summary.merge([merged_FC_weights_mainQN_summary, merged_cnn_weights_summary])


with tf.name_scope("Q_network_grads") as Q_network_gradient_summaries:
    with tf.name_scope("fully_connected_layers_grad") as mainQN_fully_connected_layers_grad_summaries:
        with tf.name_scope("original_grads") as mainQN_original_grad_summaries:
            mainQN_orig_grad_summary_list = []
            for grad, var in mainQN.gvs_Q:
                mainQN_orig_grad_summary_list.append(tf.summary.histogram(var.name + '_gradient', grad))
            merged_orig_grad_mainQN_summary = tf.summary.merge(mainQN_orig_grad_summary_list)
        with tf.name_scope("capped_grads") as mainQN_capped_grad_summaries:
            mainQN_capped_grad_summary_list = []
            for grad, var in mainQN.capped_gvs_Q:
                mainQN_capped_grad_summary_list.append(tf.summary.histogram(var.name + '_gradient', grad))
            merged_capped_grad_mainQN_summary = tf.summary.merge(mainQN_capped_grad_summary_list)
    with tf.name_scope("CNN_grad") as mainQN_cnn_grad_summaries:
        with tf.name_scope("original_grads") as mainQN_cnn_original_grad_summaries:
            mainQN_cnn_orig_grad_summary_list = []
            for grad, var in mainQN.gvs_Q:
                mainQN_cnn_orig_grad_summary_list.append(tf.summary.histogram(var.name + '_gradient', grad))
            merged_cnn_orig_grad_mainQN_summary = tf.summary.merge(mainQN_cnn_orig_grad_summary_list)
        with tf.name_scope("capped_grads") as mainQN_cnn_capped_grad_summaries:
            mainQN_cnn_capped_grad_summary_list = []
            for grad, var in mainQN.capped_gvs_Q:
                mainQN_cnn_capped_grad_summary_list.append(tf.summary.histogram(var.name + '_gradient', grad))
            merged_cnn_capped_grad_mainQN_summary = tf.summary.merge(mainQN_cnn_capped_grad_summary_list)
    merged_grad_mainQN_summary = tf.summary.merge([merged_orig_grad_mainQN_summary, merged_capped_grad_mainQN_summary, merged_cnn_orig_grad_mainQN_summary, merged_cnn_capped_grad_mainQN_summary])


init_all_variables = tf.global_variables_initializer()
sess = tf.Session()

if use_intrinsic_reward:
	tf_graph_file_name = './curiosity_model/tf_graphs/intrinsic_{0}'.format(str(datetime.datetime.now()).replace(' ','_'))
elif use_complete_random_agent:
	tf_graph_file_name = './curiosity_model/tf_graphs/random_{0}'.format(str(datetime.datetime.now()).replace(' ','_'))
else: 
	tf_graph_file_name = './curiosity_model/tf_graphs/egreedy_{0}'.format(str(datetime.datetime.now()).replace(' ','_'))

if not os.path.exists(tf_graph_file_name):
    os.makedirs(tf_graph_file_name)
writer_op_complete_Network = tf.summary.FileWriter(tf_graph_file_name, sess.graph)
sess.run(init_all_variables)


curr_episode_total_reward_summary = tf.Summary()
if load_model == True:
    print('Loading Model...')
    ckpt_complete_Network = tf.train.get_checkpoint_state(path_Complete_Network)
    saver.restore(sess, ckpt_complete_Network.model_checkpoint_path)

start_time = time.time()


# For loop for the playing of game 

print('running for {0} episodes'.format(num_episodes))

#max_steps = 100 # Not really sure if there should be max

for episode in range(num_episodes):
    episodeBuffer = experience_buffer()

    obs_t = env.reset()
    obs_t = rgb2gray(obs_t) / 255.0
    obs_t = obs_t.flatten()
    done = False 
    episode_reward = 0
    obs_tm1 = np.zeros_like(obs_t)
    obs_tm2 = np.zeros_like(obs_t)
    obs_tm3 = np.zeros_like(obs_t)
    while not done:
        env.render()
	    # Action Choice
        if use_complete_random_agent:
            a = env.action_space.sample()
        elif use_intrinsic_reward:
            a, = sess.run([mainQN.predict], feed_dict={mainQN.flattened_image: [np.array([obs_t,obs_tm1,obs_tm2,obs_tm3]).flatten()]})
            #a = a[0]

        else:
            if(np.random.rand(1) < e or total_steps < pre_train_steps): # Choose an action by greedily (with e chance of random action) from the Q-network
                a = env.action_space.sample()
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.flattened_image: [np.array([obs_t,obs_tm1,obs_tm2,obs_tm3]).flatten()]})[0]
                print(a)
        # take action, get next state
        obs_tp1, reward, done, info = env.step(a)
        obs_tp1 = rgb2gray(obs_tp1)/255.0
        obs_tp1 = obs_tp1.flatten()
        
        # save into buffer
        episodeBuffer.add(np.reshape(np.array([np.array([obs_t,obs_tm1,obs_tm2,obs_tm3]).flatten(), a, reward, obs_tp1, done]), [1, 5]))
        episode_reward += reward

        if total_steps > pre_train_steps:
            # NOTE::: We are reducing the epsilon of exploration after every action we take, not after every episode, so the epsilon decreases within 1 episode
            if e > endE:
                e -= stepDrop
        obs_tm3 = obs_tm2
        obs_tm2 = obs_tm1
        obs_tm1 = obs_t
        obs_t = obs_tp1



    print('Episode {0} with total reward {1}'.format(episode, episode_reward))
    if use_intrinsic_reward and not myBuffer.is_empty():
        Compressor_n_batches = len(episodeBuffer.buffer)//batch_size_deconv_compressor
        avg_batch_compressor_loss = 0.0
        avg_batch_intrinsic_reward = 0.0
        for i in range(Compressor_n_batches):
            curr_batch = episodeBuffer.buffer[i*batch_size_deconv_compressor:(i+1)*batch_size_deconv_compressor]
            curr_batch = np.reshape(np.array(curr_batch), [len(curr_batch), 5])
            curr_batch_state_features = np.vstack(curr_batch[:, 0])
            curr_batch_actions = curr_batch[:, 1]
            curr_batch_states_tp1 = np.vstack(curr_batch[:, 3])

            sample_, _ = myBuffer.sample(historical_sample_size)
            state_feature_sample = np.vstack(sample_[:,0])
            # action_sample = np.vstack(sample_[:,1]).astype(np.uint8)
            action_sample = sample_[:,1]
            state_tp1 = np.vstack(sample_[:,3])

            pred_tm1 = curiosity.predict_next_state(sess, state_feature_sample, action_sample)

            l = curiosity.train(sess,curr_batch_state_features, curr_batch_actions, curr_batch_states_tp1, writer_op_complete_Network, counter = episode*Compressor_n_batches+i+1)

            pred_t = curiosity.predict_next_state(sess,state_feature_sample,action_sample)
            intrinsic_r = curiosity.get_reward(predictions_t=pred_t,predictions_tm1=pred_tm1,targets=state_tp1)
            intrinsic_r = intrinsic_r * intrinsic_reward_rescaling_factor

            # save samples 
            #for i in range(10):
            #    cur_img = pred_t[i]
            #    misc.imsave(('frames/deconv_{0}.png'.format(i)),cur_img.reshape(84,84,3)*255)
            curr_batch = episodeBuffer.buffer[i * batch_size_deconv_compressor:(i + 1) * batch_size_deconv_compressor]
            for list_index in range(len(curr_batch)):
                curr_batch[list_index][2] += intrinsic_r
            #print('intrinsic Reward {0}'.format(intrinsic_r))
            avg_batch_compressor_loss += float(l)/Compressor_n_batches
            avg_batch_intrinsic_reward += float(intrinsic_r)/Compressor_n_batches
        print('compressor loss {0}'.format(avg_batch_compressor_loss))
        summary_val_compressor_loss, = sess.run([avg_batch_compressor_loss_summary], feed_dict={avg_batch_compressor_loss_placeholder: avg_batch_compressor_loss})
        writer_op_complete_Network.add_summary(summary_val_compressor_loss, episode_num + 1)
        summary_val_intrinsic_reward, = sess.run([avg_batch_intrinsic_reward_summary], feed_dict={avg_batch_intrinsic_reward_placeholder: avg_batch_intrinsic_reward})
        writer_op_complete_Network.add_summary(summary_val_intrinsic_reward, episode_num + 1)

    if total_steps > pre_train_steps:
        if (episode_num % (update_freq_per_episodes) == 0 and episode_num > 0):
            avg_batch_DQN_loss = 0.0
            Q_n_batches = 10
            for batch_num in range(Q_n_batches):
                trainBatch, actual_sampled_size = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                # Below we perform the Double-DQN update to the target Q-values
                Q1 = sess.run(mainQN.predict, feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 3])})
                Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.flattened_image: np.vstack(trainBatch[:, 3])})
                # NOTE ::: the use of end_multiplier --- the is_terminal_flag gets stored as 1 or 0(True or False),
                # NOTE ::: Done if the is_terminal_flag is true i.e 1, we define the end_multiplier as 0, if the is_terminal_flag is false i.e 0, we define the end_multiplier as 1
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(actual_sampled_size), Q1]
                targetQ = trainBatch[:, 2] + (gamma_discount_factor * doubleQ * end_multiplier)
                # Update the network with our target values.
                # NOTE ::: it is important to recalculate the Q values of the states in the experience replay and then get the gradient w.r.t difference b/w recalculated values and targets
                # NOTE ::: otherwise it defeats the purpose of experience replay, also we are not storing the Q values for this reason
                _,_,l, summary_val_grad_mainQN, summary_val_weights_mainQN = sess.run([mainQN.train_Q_op,mainQN.train_cnn_op,mainQN.loss, merged_grad_mainQN_summary, merged_weights_mainQN_summary],
                                                                                      feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})
                _ = sess.run(grouped_target_network_update_op)
                avg_batch_DQN_loss += float(l)/Q_n_batches
                writer_op_complete_Network.add_summary(summary_val_grad_mainQN, episode_num*Q_n_batches+batch_num+1)
                writer_op_complete_Network.add_summary(summary_val_weights_mainQN, episode_num * Q_n_batches + batch_num + 1)
            print('Qloss {0}'.format(avg_batch_DQN_loss))
            summary_val_DQN_loss, = sess.run([avg_batch_DQN_loss_summary], feed_dict={avg_batch_DQN_loss_placeholder: avg_batch_DQN_loss})
            writer_op_complete_Network.add_summary(summary_val_DQN_loss, episode_num + 1)
            summary_val_input_images, summary_val_cnn_merged = sess.run([input_frame_summary, cnn_merged_summaries], feed_dict={mainQN.flattened_image: np.vstack(trainBatch[:5, 3])})
            writer_op_complete_Network.add_summary(summary_val_input_images, episode_num + 1)
            writer_op_complete_Network.add_summary(summary_val_cnn_merged, episode_num + 1)

    myBuffer.add(episodeBuffer.buffer)

    summary_val_curr_episode_reward, = sess.run([curr_episode_reward_summary], feed_dict={curr_episode_total_reward_placeholder: episode_reward})
    writer_op_complete_Network.add_summary(summary_val_curr_episode_reward, episode + 1)
    reward_per_episode_list.append(episode_reward)
    
    # Periodically save the model.
    if(episode % model_saving_freq == 0 and episode>0):
        saver.save(sess, path_Complete_Network + '/model-' + str(episode) + '.ckpt')
        print("Saved Model after episode : "+str(episode))
        


end_episode_time = time.time()
duration = end_episode_time-start_time
duration = datetime.timedelta(seconds=duration)
print('Total running time is {0}'.format(duration))
saver.save(sess, path_Complete_Network + '/model-' + str(episode_num) + '.ckpt')
writer_op_complete_Network.close()


reward_per_episode_list = np.array(reward_per_episode_list)
rMean = np.average(reward_per_episode_list)
print('Mean reward is '+str(rMean))


if use_intrinsic_reward:
	results_file_name = './curiosity_model/intrinsic_{0}_results.pkl'.format(str(datetime.datetime.now()).replace(' ','_'))
elif use_complete_random_agent:
	results_file_name = './curiosity_model/random_resutls.pkl'
else:
	results_file_name = './curiosity_model/egreedy_{0}_results.pkl'.format(str(datetime.datetime.now()).replace(' ','_'))
fp = open(results_file_name, 'wb')
results_dict = {'reward_per_episode_list':reward_per_episode_list, 'mean_reward_per_episode_window_list':mean_reward_per_episode_window_list, 'steps_taken_per_episode_list':steps_taken_per_episode_list}
pickle.dump(results_dict, fp)
fp.close()

plt.plot(reward_per_episode_list)
















