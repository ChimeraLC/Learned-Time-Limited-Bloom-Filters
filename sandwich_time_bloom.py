import math
from bitarray import bitarray
from sklearn.utils import murmurhash3_32
import random

# Seed randomness
random.seed(17)

"""
Default age partitioned bloom filter
g: The number of elements contained in each generation
l: The number of generations each element will be stored for
k: Parameter mainly used to control fp rate
model: Binary classification model for if given element is 'likely' to be included
"""
class Learned_AP_Bloom():
    # Initialize bloom filter
    def __init__(self, k, l, g, model):
        # Check validity of parameters
        if k <= 0 or l <= 0 or g <= 0:
            print("Invalid parameters")
            quit()
        
        # Store and calculate initial parameters
        self.k = k  # Updated slices
        self.l = l  # Saved slices
        self.g = g  # Insertions per generation
        self.h = k + l # Current slices
        self.m = int(math.ceil(float(k * g) * math.log(math.e, 2))) # Slice bits
        self.m2 = int(self.m / 8)    # Lower slice bits
        self.cur_slice = 0 # Current slice
        self.count = 0 # Total insertions
        self.hash_mappings = [i for i in range(k+l)] # Mapping of indexes to hash functions
        self.hash_mappings_lower = [k+1+i for i in range(k+l)]  # Hash functions for backup filter
        # Create bit array
        self.bits = self.h * self.m * bitarray('0')
        # Backup bit array
        self.bits2 = self.h * self.m2 * bitarray('0')
        # Store model
        self.model = model

    # Helper function to update bit array
    def update_bits(self,  item):
        miss = self.model.classify(item) == 0   # Only classify item once
        for i in range(self.cur_slice, self.cur_slice + self.k):    # Update k most recent slices
            cur_index = i % self.h # Cyclic bit array
            # Calculate corresponding hash values
            hash_val = murmurhash3_32(item, self.hash_mappings[cur_index], True) % self.m
            self.bits[cur_index * self.m + hash_val] = 1 # Set bit to 1
            # Additionally add to backup filter if model 'misses'
            if miss:
                backup_hash = murmurhash3_32(item, 
                    self.hash_mappings_lower[cur_index], True) % self.m2
                self.bits2[cur_index * self.m2 + backup_hash] = 1


    # Helper function to update slices / generations
    def shift(self):
        self.count = 0 # Reset count
        # Clear newest slice
        new_slice = (self.cur_slice + self.k) % (self.h)
        for i in range(new_slice * self.m, (new_slice + 1) * self.m):
            self.bits[i] = 0
        # Also clear backup array
        for i in range(new_slice * self.m2, (new_slice + 1) * self.m2):
            self.bits2[i] = 0
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
        backup_matching = 0 # Matches in the backup array
        # Check if there is a contiguous k slices that include the item
        i = self.h
        while i >= self.k - matching:
            # Get the current slice
            cur_index = ((self.cur_slice + self.k - 1 + i) % self.h)
            # Calculate corresponding hash vlues
            hash_val = murmurhash3_32(item, self.hash_mappings[cur_index], True) % self.m
            backup_hash = murmurhash3_32(item, self.hash_mappings_lower[cur_index], True) % self.m2
            # Get correponting bits
            bit = self.bits[cur_index * self.m + hash_val]
            backup_bit = self.bits2[cur_index * self.m2 + backup_hash]
            # If theres a match
            if bit == 1:
                matching += 1
                if backup_bit == 1: # Matches in the backup array should only match in front array
                    backup_matching += 1
                if matching == self.k: # If there are k consecutive matches
                    # Check model
                    if self.model.classify(item) == 1:
                        return True
                    # Otherwise, check backup to prevent false negative
                    if backup_matching == self.k:
                        return True
                    # If not matching, potentially ran into a false positive
                    i -= 1
                    matching -= 1
                    continue
            else:
                backup_matching == 0
                matching = 0 # Otherwise, reset consecutive count
            i -= 1
        # Match was not found
        return False