import tensorflow as tf
from tensorflow.keras import regularizers

import numpy as np
from datetime import datetime
from collections import deque
import random
import os
from tqdm import tqdm

import SnakeEnv
from SnakeEnv import Snake
#from ModifiedTensorBoard import ModifiedTensorBoard
##############################################
#Handel Error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
gpu_memory_fraction = 0.3 # Choose this number through trial and error
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,)
session_config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=session_config, graph=graph)
'''
# For more repetitive results
'''
random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)
'''
# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
##############################################
def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def softmax_with_zero(x):
	i = np.argmin(x)
	x = np.concatenate([x[:i] , x[i+1:]])
	x = softmax(x)
	x = np.concatenate([x[:i],np.array([0]),x[i:]])
	return x

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
SNAKE_FIELD_SHAPE = (SnakeEnv.n_cells_x,SnakeEnv.n_cells_y)
INPUT_SHAPE = (SNAKE_FIELD_SHAPE[0],SNAKE_FIELD_SHAPE[1],1)
SNAKE_DIRECTION_CHOICE = 4
MINIBATCH_SIZE =64
DISCOUNT = 1
UPDATE_TARGET_EVERY = 5

EPISODES = 10_000

#exploration
epsilon = 1
EPSILON_DECAY = 1#0.9975
MIN_EPSILON = 0.25

#tensorboard settings
AGGREGATE_STATS_EVERY = 50
#Render preview, set to see the display
SHOW_PREVIEW = False

#minimum reward to save
MIN_REWARD = 0

DUMP_ACC = 0.75

class DQNAgent():
	def __init__(self,ids,model=None):
		self.NAME = "Agent-{}".format(ids)
		#self.tensorboard = ModifiedTensorBoard(log_dir="./logs/{}".format(self.NAME))

		#Env
		self.env = Snake(np.array(SNAKE_FIELD_SHAPE))

		#model which gets trained
		if model == None :
			self.model = self.create_model()
		else:
			self.model = model

		#target model which we try to predict
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		self.target_update_counter = 0

	def create_model(self):
		class Shape1(tf.keras.layers.Layer):
			def __init__(self,shape):
				super(Shape1, self).__init__()
				self.made_up_shape = shape

			def call(self, inputs):
				x = tf.reshape(inputs,[-1]+self.made_up_shape)
				x = tf.transpose(x,perm=[0,2,1])
				x = tf.reshape(x,[-1,np.prod(self.made_up_shape),1])
				return x
		
		model = tf.keras.models.Sequential()
		#model.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 1), input_shape=INPUT_SHAPE,activation='relu'))
		#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
		#model.add(tf.keras.layers.Flatten(input_shape=INPUT_SHAPE))
		#model.add(Shape1([10*10,16]))
		#model.add(tf.keras.layers.LocallyConnected1D(16,kernel_size=16,strides=16,activation=tf.nn.relu))
		model.add(tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(64))#,activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
		#model.add(tf.keras.layers.Dense(32))#,activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
		model.add(tf.keras.layers.Dense(16))#,activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
		model.add(tf.keras.layers.Dense(SNAKE_DIRECTION_CHOICE,activation=tf.nn.softmax))

		model.compile(optimizer=tf.keras.optimizers.Adam(),
					  #loss='mse',
					  loss='mse',
					  metrics=['accuracy']
					  )
		return model

	def update_replay_memory(self,transition):
		self.replay_memory.append(transition)

	def get_qs(self,state):
		return self.model.predict(np.array(state).reshape(-1,*state.shape))[0]

	def train(self,terminal_state):
		global DUMP_ACC
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return
	
		minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)

		current_states = np.array([transition[0] for transition in minibatch])
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		y = []

		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			#before = np.array(current_qs_list[index])
			current_qs = current_qs_list[index]
			if not done:
				furture_q = np.array(future_qs_list[index])
				max_future_q_index = np.argmax(furture_q)
				new_q = reward + DISCOUNT * furture_q[max_future_q_index]
				current_qs[np.argmax(action)] = new_q

			if done:
				new_q = 0
				current_qs[np.argmax(action)] = new_q

			X.append(current_state)
			y.append(current_qs)

		#print('Before',before,'After',current_qs)
		self.model.fit(np.array(X),np.array(y),batch_size=MINIBATCH_SIZE,
			verbose=1, shuffle=False)#, callbacks=[self.tensorboard] if terminal_state else None)

		# Upating to determine if we want to update target model
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

			#purge good outcome samples
			current_states = np.array([transition[0] for transition in self.replay_memory])
			current_qs_list = self.model.predict(current_states)

			new_current_states = np.array([transition[3] for transition in self.replay_memory])
			future_qs_list = self.target_model.predict(new_current_states)

			index_list = []

			X = []
			y = []

			for index, (current_state, action, reward, new_current_state, done) in enumerate(self.replay_memory):
				current_qs = current_qs_list[index]
				if not done:
					furture_q = np.array(future_qs_list[index])
					max_future_q_index = np.argmax(furture_q)
					new_q = reward + DISCOUNT * furture_q[max_future_q_index]
					current_qs[np.argmax(action)] = new_q
					current_qs = softmax(current_qs)

				if done:
					new_q = 0
					current_qs[np.argmax(action)] = new_q
					current_qs = softmax_with_zero(current_qs)

				X.append(current_state)
				y.append(current_qs)
			
			loss,acc = self.target_model.evaluate(np.array(X),np.array(y),verbose=0)
			if acc > DUMP_ACC:
				print(loss,acc)
				self.replay_memory.clear()




if __name__ == '__main__':

	import sys
	if len(sys.argv) < 2:
		from glob import glob
		pre_trained_models = sorted(glob('models/Agent*'))
	else:
		pre_trained_models = ['models/'+sys.argv[1]]
	model = None if len(pre_trained_models) == 0 else tf.keras.models.load_model(pre_trained_models[-1])

	agent = DQNAgent(1,model)
	# For stats
	ep_rewards = [agent.env.get_reward()]

	for episode in tqdm(range(1,EPISODES+1),ascii=True,unit='episode'):
		#agent.tensorboard.step = episode

		episode_reward = 0
		current_state = agent.env.reset()

		done = False

		while not done:
			if np.random.random() > epsilon:
				action = agent.get_qs(current_state)
			else:
				num_actions = len(agent.env.DIRECTION_ARRAY)
				action = [0]*num_actions
				
				while True:
					act = np.random.randint(0,num_actions)
					action[act] = 1 #
					break # Remove to explore better
					if act != agent.env.reverse_dir():
						action[act] = 1
						break

			running = agent.env.update_with_one_hot(action)
			done = not running
			new_state = agent.env.get_state()
			reward   = agent.env.get_reward()

			episode_reward += reward

			if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
				# Need to add render code, Not possible in container
				pass

			agent.update_replay_memory((current_state,action,reward,new_state,done))
			agent.train(done)

			current_state = new_state
		
		# Append episode reward to a list and log stats (every given number of episodes)
		ep_rewards.append(episode_reward)
		if not episode % AGGREGATE_STATS_EVERY:
			average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
			min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
			max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
			#agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
			print('Avg:',average_reward,'Max:',max_reward)

			# Save model, but only when min reward is greater or equal a set value
			if average_reward > MIN_REWARD:
				agent.model.save(f'models/{agent.NAME}_{datetime.now().strftime("%m:%d:%Y_%H:%M:%S")}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_.model')
				MIN_REWARD = average_reward

		# Decay epsilon
		if epsilon > MIN_EPSILON:
			epsilon *= EPSILON_DECAY
			epsilon = max(MIN_EPSILON, epsilon)

	agent.model.save(f'models/temp')