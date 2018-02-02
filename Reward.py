import tensorflow as tf
import numpy as np

import Perception 




class Compressor:

	def __init__(self,frame_height, frame_width, frame_channels, num_stacked_frames,
				state_feature_size=None, total_num_actions=None, CNN_W=[], network_name=''):

		self.CNN_w = CNN_W
		self.flat_image = tf.placeholder(tf.float32,shape=[None,frame_height*frame_width*frame_channels*num_stacked_frames],
											name='Compressor_state_input')
		#self.image = tf.reshape(self.image,[-1,frame_height,frame_width,frame_channels])
		#self.state_feature = tf.placeholder(tf.float32,shape=[None,state_feature_size],
		#									name='Compressor_state_input')
		self.action = tf.placeholder(tf.uint8,shape=[None],name='Compressor_action_input')
		self.action_one_hot = tf.one_hot(self.action, total_num_actions, dtype=tf.float32)
		self.state_tp1 = tf.placeholder(tf.float32,shape=[None,frame_height*frame_width*frame_channels],
										name='Compressor_state_tp1_input')

		# self.state_feature, self.CNN_w = Perception.CNN(input=self.flat_image,height=frame_height,width=frame_width,in_channel=frame_channels,out_channel=32,weights=CNN_W)
		self.state_feature, _ = Perception.CNN(input=self.flat_image,height=frame_height,width=frame_width,in_channel=frame_channels*num_stacked_frames,out_channel=state_feature_size,weights=self.CNN_w)

		self.predicted_image, self.compressor_weights = Perception.Predictor(state=self.state_feature,
												state_size=state_feature_size,
												action=self.action_one_hot,
												action_space=total_num_actions,
												height=frame_height,
												width=frame_width,
												out_channel=frame_channels,
												network_name='Compressor')

		self.prediction_flattened = tf.contrib.layers.flatten(self.predicted_image)

		self.predictor_loss = tf.reduce_mean(tf.square(self.state_tp1 - self.prediction_flattened))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, name='compressor_adam_opt')
		self.gvs_dcnn = self.optimizer.compute_gradients(self.predictor_loss,var_list=self.compressor_weights)
		#if gvs == None:
		#	print ('helll oooo ooo ======')
		# capped_gvs = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in gvs]
		self.capped_gvs_dcnn = self.gvs_dcnn
		self.train_pred = self.optimizer.apply_gradients(self.capped_gvs_dcnn, name='compressor_weights_grad_update')

		self.gvs_cnn = self.optimizer.compute_gradients(self.predictor_loss,var_list=self.CNN_w)
		self.train_cnn = self.optimizer.apply_gradients(self.gvs_cnn, name='CNN_weights_compressor_grad_update')


		with tf.variable_scope("Compressor_grads") as compressor_grad_summaries:
			with tf.variable_scope("fully_connected_layers_grad") as compressor_fully_connected_layers_grad_summaries:
				with tf.variable_scope("original_grads") as compressor_original_grad_summaries:
					compressor_orig_grad_summary_list = []
					for grad, var in self.gvs_dcnn:
						compressor_orig_grad_summary_list.append(tf.summary.histogram(var.name + '_gradient', grad))
					self.merged_orig_grad_compressor_summary = tf.summary.merge(compressor_orig_grad_summary_list)
			with tf.variable_scope("CNN_grad") as compressor_cnn_grad_summaries:
				with tf.variable_scope("original_grads") as compressor_cnn_original_grad_summaries:
					compressor_cnn_orig_grad_summary_list = []
					for grad, var in self.gvs_cnn:
						compressor_cnn_orig_grad_summary_list.append(tf.summary.histogram(var.name + '_gradient', grad))
					self.merged_cnn_orig_grad_compressor_summary = tf.summary.merge(compressor_cnn_orig_grad_summary_list)
			self.merged_grad_compressor_summary = tf.summary.merge([self.merged_orig_grad_compressor_summary, self.merged_cnn_orig_grad_compressor_summary])

		with tf.name_scope("Compressor_weights") as compressor_weights_summaries:
			compressor_weights_summary_list = []
			for var in self.compressor_weights:
				compressor_weights_summary_list.append(tf.summary.histogram(var.name, var))
			self.merged_compressor_weights_summary = tf.summary.merge(compressor_weights_summary_list)

	def trainDCNN(self,sess,states, actions, states_tp1, writer_op_complete_Network, counter = None):
		_, loss, summary_val_grad_compressor = sess.run([self.train_pred, self.predictor_loss, self.merged_orig_grad_compressor_summary],
						feed_dict={self.flat_image:states,self.action:actions,
								self.state_tp1:states_tp1})
		writer_op_complete_Network.add_summary(summary_val_grad_compressor, counter)
		return loss

	def train(self,sess,states, actions, states_tp1, writer_op_complete_Network, counter = None):
		_, _, loss, summary_val_grad_compressor, summary_val_weights_compressor = sess.run([self.train_pred, self.train_cnn, self.predictor_loss, self.merged_grad_compressor_summary, self.merged_compressor_weights_summary],
						feed_dict={self.flat_image:states,self.action:actions,
							self.state_tp1:states_tp1})
		writer_op_complete_Network.add_summary(summary_val_grad_compressor, counter)
		writer_op_complete_Network.add_summary(summary_val_weights_compressor, counter)
		return loss


	def predict_next_state(self,sess,states,actions):
		prediction_flattened_value, = sess.run([self.prediction_flattened],
			feed_dict={self.flat_image:states,self.action:actions})
		return prediction_flattened_value


	def get_reward(self,predictions_t,predictions_tm1,targets):
		loss_tm1 = np.mean(np.square(predictions_tm1 - targets))
		loss_t = np.mean(np.square(predictions_t - targets))
		improvement = (loss_tm1 - loss_t)
		if improvement < 0 :
			return 0
		return improvement






























