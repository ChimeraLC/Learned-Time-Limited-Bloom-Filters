import math
from bitarray import bitarray
from sklearn.utils import murmurhash3_32
import random

# Seed randomness
random.seed(17)
# Hash function factory that, when given hash table size m, returns  hash function mapping integers to [m]
def hashfunc(m):
        # Generate random values
        seed = random.randint(0, 2048)
        # Return corresponding murmurhash
        def hashf(i):
                return murmurhash3_32(i, seed, True) % m

        # Return function
        return hashf

class Bloom_Filter():
        # Initialize the bloom filter with n expected keys, false positive rate fp_rate
        def __init__(self, n, fp_rate):
                # Calculating required size
                r = n * math.log(fp_rate, 0.618)
                # Finding corresponding k
                k = (int) (r / n * math.log(2))
                # Find smallest power of 2 greater than r
                ceil_pow = math.ceil(math.log(r)/math.log(2))
                # Create bit array
                self.bits = 2 ** ceil_pow * bitarray('0')
                self.hashes = []
                # Get corresponding has functions
                for _ in range(k):
                        self.hashes.append(hashfunc(2 ** ceil_pow))

        def insert(self, key):
                # Set all k bits
                for hash in self.hashes:
                        self.bits[hash(key)] = 1

        def test(self, key):
                included = True
                for hash in self.hashes:
                        if self.bits[hash(key)] == 0:
                                included = False
                return included

# Membership and testing sets (unique)
values = random.sample(range(10000,100000), 11000)
members = values[0:10000]
# Select 1000 nonmembers
outside = values[10000:]
# Select 1000 members
inside = [members[i] for i in random.sample(range(0, 10000), 1000)]

#0.01 Bloom Filter
b_filter = Bloom_Filter(10000, 0.01)
# Inserting
for val in members:
        b_filter.insert(val)
# Checking false positives and correct positives
fp = 0
cp = 0
for i in range(1000):
        if b_filter.test(outside[i]):
                fp += 1
        if b_filter.test(inside[i]):
                cp += 1
print("The 0.01 Bloom Filter had a false positive rate of:",fp/1000,
       "while", cp, "of the members were properly detected as included.")


#0.01 Bloom Filter
b_filter = Bloom_Filter(10000, 0.001)
# Inserting
for val in members:
        b_filter.insert(val)
# Checking false positives and correct positives
fp = 0
cp = 0
for i in range(1000):
        if b_filter.test(outside[i]):
                fp += 1
        if b_filter.test(inside[i]):
                cp += 1
print("The 0.001 Bloom Filter had a false positive rate of:",fp/1000,
       "while", cp, "of the members were properly detected as included.")


#0.01 Bloom Filter
b_filter = Bloom_Filter(10000, 0.0001)
# Inserting
for val in members:
        b_filter.insert(val)
# Checking false positives and correct positives
fp = 0
cp = 0
for i in range(1000):
        if b_filter.test(outside[i]):
                fp += 1
        if b_filter.test(inside[i]):
                cp += 1
print("The 0.0001 Bloom Filter had a false positive rate of:",fp/1000,
       "while", cp, "of the members were properly detected as included.")