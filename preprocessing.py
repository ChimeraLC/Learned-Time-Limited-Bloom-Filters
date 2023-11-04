import numpy as np
import pandas as pd
import string
import unicodedata

from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split


class Preprocessing:
	
	def __init__(self, args):
		self.valid_path = args.valid_data_path
		self.invalid_path = args.invalid_data_path
		self.max_len = args.max_len	# Max length of each username
		self.test_percentage = args.train_ratio
		# Characters within each username
		self.chars = args.chars

	def load_data(self):
		# Get all player data
		self.validData = pd.read_csv(self.valid_path)
		self.validData.drop(['rank','total','attack','defence','strength','hitpoints',
				  'ranged','prayer','magic','cooking','woodcutting','fletching',
				  'fishing','firemaking','crafting','smithing','mining','herblore',
				  'agility','thieving','slayer','farming','runecraft','hunter',
				  'construction'], axis=1, inplace=True)
		
		# Get false usernames
		self.invalidData = pd.read_csv(self.invalid_path)
		
		# Get username data
		xValid = self.validData['username']
		xInvalid = self.invalidData['username']

		# Combining datasets
		X = np.append(xValid, xInvalid)
		# 0 represents invalid username, while 1 represents valid
		Y = pd.DataFrame(data = [1 for i in range(len(xValid))] + [0 for i in range(len(xInvalid))], dtype = np.float32)
		
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