"""
Project 4 - Recurrent Neural Networks with Tensorow and Keras
Course: COSC 525: Deep Learning (Spring 2020)
Authors: Haoyuan Sun, Ximu Zhang
Date: 04/15/2020
"""
# library import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN
import sys


def save_data(data_file_name, data):
	"""
	Function to save data.
	:param data_file_name: name of the data file
	:param data: data used
	:return: none
	"""
	f = open(data_file_name, 'w')
	f.write(str(data))
	f.close()


def load_data(data_file_name):
	"""
	Function to load saved data file.
	:param data_file_name: name of the data file
	:return: data used
	"""
	f = open(data_file_name, 'rb')
	data = 0  # TODO: load data
	f.close()
	return data


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
		for text in lines:  # attach lines togeter
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
	save_data('train_set.txt', training_data)
	return training_data


def model_train(model_name, data):
	"""
	Function to proceed the model training.
	:param model_name: chosen model
	:param data: training data
	:return: none
	"""
	model = Sequential()
	if model_name == 'simple_rnn':
		model.add(SimpleRNN(units=100, input_dim=58))
		model.add(Dense(58))
	elif model_name == 'lstm':
		model.add(LSTM(100, input_dim=58, return_sequences=True))
		model.add(Dense(58))
	else:
		print("Error model chosen.")
		sys.exit(-1)


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
	train_set = data_processing(file_name, window_size, stride)

	# train model
	model_train(model_select, train_set)
