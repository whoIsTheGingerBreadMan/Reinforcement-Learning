from tensorflow.python import keras
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from utils.pre_process import convert_uint8_rgb_to_unit_gray
import copy
import random
from exploration_strategies.linear import LinearEpsilon
from collections import deque

MAX_MEMORY = 30000
BATCH_SIZE= 5

class Agent:
    def __init__(self,env = None):
        if env:
            self.set_environment(env)
        else:
            self.env = None

    def set_environment(self,env):
        self.env = env

    def _step(self,action):
        return self.env.step(action)

    def reset_environment(self):
        return self.env.reset()


class RandomAgent(Agent):
    def __init__(self,env=None):
        super()
    def take_action(self):
        return self.env.action_space.sample()


class DQN(Agent):

    def __init__(self,env=None,observation_size=[84,84],Q_network_instantiator=None,gamma=.99,epsilon=.3,C=100):
        self.k = 5
        Agent.__init__(self,env)
        self.observation_size = [1,observation_size[0],observation_size[1],self.k]
        self.set_Q_Network()
        self.gamma = gamma
        self.optimizer = tf.optimizers.Adam()
        self.steps = 1
        self.C = C
        self.epsilon = LinearEpsilon()
        self.current_observation = np.zeros(self.observation_size,"float32")

    def set_Q_Network(self):
        if not self.env:
            print("need to set env first. Use set_environment(env)")
            return
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32,(8,8),(4,4),padding="same",activation="relu",input_shape=self.observation_size[1:]))
        model.add(keras.layers.Conv2D(64,(4,4),(2,2),padding="same",activation="relu"))
        model.add(keras.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation="relu"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512,"relu"))
        model.add(keras.layers.Dense(self.env.action_space.n, "relu",dtype="float32"))
        self.predict = model
        self.target = copy.copy(model)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.losses = {"predict":[],"target":[]}
        print("made model")

    def set_observation_size(self,new_observation_size):
        self.observation_size = new_observation_size

    def set_environment(self,env):
        self.env = env
        self.set_Q_Network()

    def get_next_observation(self,action):
        tot_reward = 0
        tot_done = False
        full_observations =np.zeros(self.observation_size,dtype="float32")
        for i in range(self.k):
            (next_observation, reward, done, info) = self._step(action)
            tot_reward += reward
            tot_done = tot_done or done
            if tot_done:
                self.env.reset()
                break
            observation = convert_uint8_rgb_to_unit_gray(next_observation,self.observation_size[1:3])
            full_observations[0,:,:,i] = observation
        return full_observations,tot_reward,tot_done


    def step(self,current_observation,reset=False):
        if reset:
            observation = self.env.reset()
            return observation

        if random.uniform(0,1)<self.epsilon.get_epsilon():
            action = self.env.action_space.sample()
        else:
            Q_values = self.predict(self.current_observation)
            action= np.argmax(Q_values)

        self.previous_observation = self.current_observation
        self.current_observation,reward,done = self.get_next_observation(action)

        self.steps += 1
        self.memory.append([self.current_observation,action,reward,self.previous_observation])

        if self.steps>BATCH_SIZE:
            self.train_predict()
        if self.steps % self.C == 0:
            self.train_target()


    def train_target(self):
        self.target = keras.models.clone_model(self.predict)


    def train_predict(self):
        outs = random.sample(self.memory, min(len(self.memory),BATCH_SIZE))
        with tf.GradientTape() as tape:
            tape.watch(self.predict.trainable_variables)
            loss = tf.reduce_mean(
                [(r + self.gamma * np.max(self.target(s2)) - self.predict(s1)[0, a1]) ** 2 for s1, a1, r, s2 in outs])
        self.losses["predict"].append(loss)
        grads = tape.gradient(loss, self.predict.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.predict.trainable_variables))
