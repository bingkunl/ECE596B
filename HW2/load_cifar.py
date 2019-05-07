import pickle
import numpy as np

#Step 1: define a function to load traing batch data from directory
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def load_training_batch(folder_path,batch_id):
	"""
	Args:
		folder_path: the directory contains data files
		batch_id: training batch id (1,2,3,4,5)
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

	###load batch using pickle###
	file = folder_path + "\\data_batch_" + str(batch_id)
	dict = unpickle(file)
	###fetch features using the key ['data']###
	features = dict[b'data']
	###fetch labels using the key ['labels']###
	labels = dict[b'labels']
	return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
	"""
	Args:
		folder_path: the directory contains data files
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

	###load batch using pickle###
	file = folder_path + "\\test_batch"
	dict = unpickle(file)
	###fetch features using the key ['data']###
	features = dict[b'data']
	###fetch labels using the key ['labels']###
	labels = dict[b'labels']
	return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
	file = folder_path + "\\batches.meta"
	dict = unpickle(file)
	return dict[b'label_names']

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
	"""
	Args:
		features: a numpy array with shape (10000, 3072)
	Return:
		features: a numpy array with shape (10000,32,32,3)
	"""
	return features.reshape(10000, 32, 32, 3)

#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
	"""
	Args:
		folder_path: directory that contains data files
		batch_id: the specific number of batch you want to explore.
		data_id: the specific number of data example you want to visualize
	Return:
		None

	Descrption: 
		1)You can print out the number of images for every class. 
		2)Visualize the image
		3)Print out the minimum and maximum values of pixel 
	"""
	pass

#Step 6: define a function that does min-max normalization on input
def normalize(x):
	"""
	Args:
		x: features, a numpy array
	Return:
		x: normalized features
	"""
	x = (x - x.min())/(x.max() - x.min())
	return x

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
	"""
	Args:
		x: a list of labels
	Return:
		a numpy array that has shape (len(x), # of classes)
	"""
	arr = np.zeros([len(x), 10])
	for i in range(len(x)):
		arr[i][x[i]] = 1
	return arr

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
	"""
	Args:
		features: numpy array
		labels: a list of labels
		filename: the file you want to save the preprocessed data
	"""
	features = normalize(features)
	arr = one_hot_encoding(labels)
	dict = {'labels': arr, 'data': features}
	file = open(filename, 'wb')
	pickle.dump(dict, file)
	file.close()

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
	"""
	Args:
		folder_path: the directory contains your data files
	"""
	validation_data = []
	validation_labels = []
	for i in range(1,6):
		data, labels = load_training_batch(folder_path, i)
		preprocess_and_save(data[:9000], labels[:9000], 'preprocessed_data_batch_' + str(i))
		validation_data.append(data[9000:])
		validation_labels.append(labels[9000:])
	validation_data = np.concatenate(validation_data)
	validation_labels = np.concatenate(validation_labels)
	preprocess_and_save(validation_data, validation_labels, 'preprocessed_validation_batch')
	test_data, test_labels = load_testing_batch(folder_path)
	preprocess_and_save(test_data, test_labels, 'preprocessed_test_batch')

#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
	"""
	Args:
		features: features for one batch
		labels: labels for one batch
		mini_batch_size: the mini-batch size you want to use.
	Hint: Use "yield" to generate mini-batch features and labels
	"""
	s = 0
	f = mini_batch_size
	while f < len(labels):
		yield (features[s:f], labels[s:f])
		s = s + mini_batch_size
		f = f + mini_batch_size

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
	"""
	Args:
		batch_id: the specific training batch you want to load
		mini_batch_size: the number of examples you want to process for one update
	Return:
		mini_batch(features,labels, mini_batch_size)
	"""
	file_name = 'preprocessed_data_batch_' + str(batch_id)
	dict = unpickle(file_name)
	features = dict['data']
	labels = dict['labels']
	return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
	file_name = 'preprocessed_validation_batch'
	dict = unpickle(file_name)
	features = dict['data']
	labels = dict['labels']
	return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
	file_name = 'preprocessed_test_batch'
	dict = unpickle(file_name)
	features = dict['data']
	labels = dict['labels']
	return mini_batch(features,labels,test_mini_batch_size)

def load_all_test_batch():
	file_name = 'preprocessed_test_batch'
	dict = unpickle(file_name)
	features = dict['data']
	labels = dict['labels']
	return features,labels

