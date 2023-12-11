import math
from bitarray import bitarray
from sklearn.utils import murmurhash3_32
import random

"""
Default age partitioned bloom filter
g: The number of elements contained in each generation
l: The number of guarenteed generations (no false negatives)
k: The number of transitory generations (possible false negatives, older)
fp: Aimed for fp rate
"""
class AP_Bloom():
    # Initialize bloom filter
    def __init__(self, k, l, g, fp):
        # Check validity of parameters
        if k <= 0 or l <= 0 or g <= 0:
            print("Invalid parameters")
            quit()
        
        # Store and calculate initial parameters
        self.k = k  # Updated slices
        self.l = l  # Saved slices
        self.g = g  # Insertions per generation
        self.h = k + l # Current slices
        # Calculate the parameters necessary for getting the desired fp
        self.m = int(self.calc_params(k, l, fp) * math.ceil(float(k * g))) # Slice bits
        self.cur_slice = 0 # Current slice
        self.count = 0 # Total insertions
        self.hash_mappings = [i for i in range(k+l)] # Mapping of indexes to hash functions

        # Create bit array
        self.bits = self.h * self.m * bitarray('0')

    # Helper function to update bit array
    def update_bits(self,  item):
        for i in range(self.cur_slice, self.cur_slice + self.k):    # Update k most recent slices
            cur_index = i % self.h # Cyclic bit array
            # Calculate corresponding hash values
            hash_val = murmurhash3_32(item, self.hash_mappings[cur_index], True) % self.m
            self.bits[cur_index * self.m + hash_val] = 1 # Set bit to 1

    # Helper function to update slices / generations
    def shift(self):
        self.count = 0 # Reset count
        # Clear newest slice
        new_slice = (self.cur_slice + self.k) % (self.h)
        for i in range(new_slice * self.m, (new_slice + 1) * self.m):
            self.bits[i] = 0
        # Shift curent slice
        self.cur_slice = (self.cur_slice + 1) % self.h

    # Inserts an element into the set
    def insert(self, item):
        # Check if we need to update generation
        if self.count == self.g:
            self.shift()    # Shift slices

        # Update bits to account for new item
        self.update_bits(item)

        # Update counts
        self.count += 1
    
    # Returns true if the item is found in the set, false otherwise, only false positives
    def query(self, item):
        matching = 0 # Consecutive matches
        # Check if there is a contiguous k slices that include the item
        i = self.h
        while i >= self.k - matching:
            # Get the current slice
            cur_index = ((self.cur_slice + self.k - 1 + i) % self.h)
            # Calculate corresponding hash vlues
            hash_val = murmurhash3_32(item, self.hash_mappings[cur_index], True) % self.m
            # Get correponting bit
            bit = self.bits[cur_index * self.m + hash_val]

            # If theres a match
            if bit == 1:
                matching += 1
                if matching == self.k: # If there are k consecutive matches
                    return True
            else:
                matching = 0 # Otherwise, reset consecutive count
            i -= 1
        # Match was not found
        return False
    
    # Returns the overall size of the filer
    def get_size(self):
        return len(self.bits) // 8
    

    # Returns how 'full' the filter is
    def get_usage(self):
        main_usage = 0
        for i in range(self.h * self.m):
            if self.bits[i] == 1:
                main_usage += 1
        print("Main filter usage:", main_usage / (self.h * self.m))

    # Returns parameters to give the desired fp (underlying math might be wrong
    def calc_params(self, k, l, fp):
        coef = math.exp(k / 2 - 1 - l/2) + math.exp(k / 3.1) - 1.8
        t = (math.log(fp, 10)) ** 2 + 0.8 * k - 1.5
        return t / coef * math.log(math.e, 2)

