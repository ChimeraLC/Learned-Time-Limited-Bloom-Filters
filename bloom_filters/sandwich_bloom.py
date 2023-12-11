from bitarray import bitarray
from sklearn.utils import murmurhash3_32
import random

def hashfunc(m):
        # Generate random values
        seed = random.randint(0, 2048)
        # Return corresponding murmurhash
        def hashf(i):
                return murmurhash3_32(i, seed, True) % m

        # Return function
        return hashf

"""
Initial implementation of a binary classification model within a bloom filter (UNUSED)
"""
class Learned_Bloom():
        # Initialize the sandwitched bloom filter with r, n, k on first level
        # Internal model model, and
        # r2, n2, k2 on second layer
        def __init__(self, r, k, r2, k2, model):
                # Create first bit array
                self.bits = r * bitarray('0')
                self.hashes = []
                # Get corresponding hash functions
                for _ in range(k):
                        self.hashes.append(hashfunc(r))
                # Create second bit array
                self.bits2 = r2 * bitarray('0')
                self.hashes2 = []
                # Get corresponding hash functions
                for _ in range(k2):
                        self.hashes2.append(hashfunc(r2))
                # Store classification model
                self.model = model
            
        def insert(self, key):
                # Set all k bits
                for hash in self.hashes:
                        self.bits[hash(key)] = 1
                # Set all k2 bits if it is declared false by model
                if (self.model.classify(key) == 0):
                    for hash in self.hashes2:
                            self.bits2[hash(key)] = 1
        
        def test(self, key):
                included = True
                for hash in self.hashes:
                        if self.bits[hash(key)] == 0:
                                included = False
                # If included, further check the model
                if included:
                    # If model confirms, return true
                    if (self.model.classify(key) == 1):
                           return True
                    # Otherwise, check backup filter
                    included = True
                    for hash in self.hashes2:
                            if self.bits2[hash(key)] == 0:
                                    included = False
                    return included
                # Otherwise, definite false
                else:
                    return False