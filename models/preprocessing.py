import numpy as np
import pandas as pd

from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split

# Preprocessing to convert username data to vectors
class Preprocessing:
	def __init__(self, args):
		self.common_path = args.common_data_path
		self.uncommon_path = args.uncommon_data_path
		self.max_len = args.max_len	# Max length of each username
		self.test_percentage = args.train_ratio
		self.data_length = args.data_length
		# Characters within each username
		self.chars = args.chars

	def load_data(self):
		# Get all player data
		self.commonData = pd.read_csv(self.common_path)
		self.commonData.drop(['rank','total','attack','defence','strength','hitpoints',
				  'ranged','prayer','magic','cooking','woodcutting','fletching',
				  'fishing','firemaking','crafting','smithing','mining','herblore',
				  'agility','thieving','slayer','farming','runecraft','hunter',
				  'construction'], axis=1, inplace=True)
		
		# Get false usernames
		self.uncommonData = pd.read_csv(self.uncommon_path)
		
		# Get username data
		xCommon = self.commonData['username'][:self.data_length]
		xUncommon = self.uncommonData['username'][:self.data_length]

		# Combining datasets
		X = np.append(xCommon, xUncommon)
		# 0 represents uncommon username, while 1 represents common
		Y = pd.DataFrame(data = [1 for i in range(len(xCommon))] + [0 for i in range(len(xUncommon))], dtype = np.float32)
		

		# Get training and testing sets
		self.test_size = int(len(X) * self.test_percentage)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size, shuffle = True)



	# Converts a string str to the corresponding labels
	def string_to_label(self, str):
		res = []
		for _, char in enumerate(str):
			res.append(self.chars.find(char) + 1)
			if (self.chars.find(char) == -1):
				print("Unexpected character found:", char)
				quit()
		return res

	# Converts a sequence of strings to sequences
	def label_sequences(self, x):
		sequences = [self.string_to_label(str) for str in x]
		return pad_sequences(sequences, maxlen=self.max_len)