"""
Project 4 - Recurrent Neural Networks with Tensorflow and Keras
Course: COSC 525: Deep Learning (Spring 2020)
Authors: Haoyuan Sun, Ximu Zhang
Date: 04/15/2020
"""
# library import
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN
import matplotlib.pyplot as plt
import sys
import numpy as np
import time


def plot_loss_figure(param_hist, param_time_callback):
	# plot the training epoch-loss and time-loss figures
	# plot epoch-loss figure
	plt.figure()
	plt.title('Epoch-Loss')
	plt.plot(param_hist.epoch, param_hist.history['loss'])
	plt.xlabel('epoch')
	plt.ylabel('loss')

	# plot time-loss figure
	# plt.figure()
	# plt.title('Time-Loss')
	# plt.plot(param_time_callback.timefromstart, param_hist.history['loss'])
	# plt.xlabel('time')
	# plt.ylabel('loss')
	plt.show()


# class definition
class TimeHistory(keras.callbacks.Callback):
	"""
	Class for recording time of each epoch in the training.
	Reference: https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during
	-model-fit
	"""

	def on_train_begin(self, logs={}):
		self.times = []
		self.timefromstart = []
		self.starttime = time.time()

	def on_train_end(self, logs={}):
		self.totaltime = time.time() - self.starttime

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)
		self.timefromstart.append(time.time() - self.starttime)


def save_data(data_file_name, data):
	"""
	Function to save data.
	:param data_file_name: name of the data file
	:param data: data used
	:return: none
	"""
	f = open(data_file_name, 'w', encoding='utf-8')
	for seq in data:
		f.write(seq + '\n')
	f.close()


# def load_data(data_file_name):
# 	"""
# 	Function to load saved data file.
# 	:param data_file_name: name of the data file
# 	:return: data used
# 	"""
# 	f = open(data_file_name, 'rb')
# 	data = 0  # TODO: load data
# 	f.close()
# 	return data


def data_processing(text_file_name, ws, st):
	"""
	Read text file and split text into sequences for training.
	:param text_file_name: name of the text file
	:param ws: window size
	:param st: stride
	:return: training data
	"""
	file_path = './' + text_file_name  # file path

	with open(file_path, 'rt', encoding='utf-8') as f:  # open the file and read
		lines = f.readlines()  # read data
		data_str = ''
		for text in lines:  # attach lines together
			data_str += text.replace('\n', '$')  # replace '\n' with '$'
		f.close()

	# split the data using sliding window
	start_ind = 0
	training_data = []
	while start_ind <= len(data_str) - ws - st:
		extract_str = data_str[start_ind: start_ind + ws + 1]
		training_data.append(extract_str)
		start_ind += st

	# save training set
	train_set_file = 'train_set.txt'
	save_data(train_set_file, training_data)
	return training_data, train_set_file


def data_processing_2(text_file_name):
	"""
	Read text file in which each line is a single training sequence, assemble input and output array for training.
	:param text_file_name: name of the text file
	:return: input and output array for training
	"""
	file_path = './' + text_file_name  # file path

	with open(file_path, 'rt', encoding='utf-8') as f:  # open the file and read
		lines = f.readlines()  # read data
		lines = [i.rstrip('\n') for i in lines]
		f.close()

	code = []
	for line in lines:
		code = list(set().union(code, np.unique(list(line))))

	m = len(lines)
	n = len(lines[0]) - 1
	p = len(code)

	# initialize array
	train_input = np.zeros((m, n, p))
	train_output = np.zeros((m, p))

	# one hot encoding
	for i in range(m):
		train_output[i][code.index(lines[i][-1])] = 1
		for j in range(n):
			train_input[i][j][code.index(lines[i][j])] = 1

	return train_input, train_output


def model_train(model_name, data, target, hidden_state):
	"""
	Function to proceed the model training.
	:param model_name: chosen model
	:param data: training data
	:param target: training target
	:param hidden_state: hidden layer size of the rnn
	:return: none
	"""
	# size of the training data and target
	m = len(data)
	n = len(data[0])
	p = len(data[0][0])

	model = Sequential()
	if model_name == 'simple_rnn':
		model.add(SimpleRNN(units=hidden_state, input_shape=(n, p)))
		model.add(Dense(p, activation='sigmoid'))
	elif model_name == 'lstm':
		model.add(LSTM(100, input_dim=58, return_sequences=True))
		model.add(Dense(58))
	else:
		print("Error model chosen.")
		sys.exit(-1)

	# training the model
	time_callback = TimeHistory()  # time callback used in model.fit()
	batch_size = 200
	epochs = 50
	model.compile(optimizer='adam',
	              loss='mse',
	              metrics=['accuracy'])
	hist = model.fit(data, target,
	                 epochs=epochs,
	                 callbacks=[time_callback])

	plot_loss_figure(hist, time_callback)


def simple_rnn():
	"""
	Function to create simple rnn model.
	:return: simple rnn model
	"""
	pass


if __name__ == "__main__":
	cmd_command = 0  # set parameters manually
	if cmd_command == 1:
		# use cmd to run the code
		file_name = sys.argv[1]
		model_select = sys.argv[2]
		hidden_state_size = sys.argv[3]
		window_size = sys.argv[4]
		stride = sys.argv[5]
	else:
		# manually set the parameters in the program
		file_name = "beatles.txt"
		model_select = "simple_rnn"
		hidden_state_size = 100
		window_size = 10
		stride = 5

	# data preprocessing
	train_set, data_file = data_processing(file_name, window_size, stride)
	x_train, y_train = data_processing_2(data_file)

	# train model
	model_train(model_select, x_train, y_train, hidden_state_size)
