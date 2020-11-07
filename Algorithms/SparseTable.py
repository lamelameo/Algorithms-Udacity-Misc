""" Simple implementation of Sparse Table data structure """
import math


# Sparse Table uses O(nlogn) space taking theta(nlogn) time but allows for O(1) queries for max in any range
class SparseTable:
    def __init__(self, array):
        self.array = array
        self.table = [[0 for _ in range(1 + int(math.log2(len(array))))] for _ in range(len(array))]
        self.construct()

    def construct(self):
        n = len(self.array)
        # run through size 1 range maxes first so we can use 2^(j-1) maxes each remaining loop
        for i in range(n):
            self.table[i][0] = self.array[i]
        # loop through ranges sized 2^j for j = 1 to floor(logn)
        for j in range(1, 1 + int(math.log2(n))):
            # calc max for size 2^j ranges (there are n - 2^j of them)
            for i in range(n - (1 << j) + 1):
                # max for this size 2^j range is max of two sized 2^(j-1) ranges calculated in previous loop
                self.table[i][j] = max(self.table[i][j - 1], self.table[i + (1 << (j - 1))][j - 1])

    def max(self, index_start, index_end):
        # get biggest power of 2 <= size of given range
        j = int(math.log2(index_end - index_start + 1))
        # calculate maximum based on the two table items of that size that cover the range
        # ie the 2 items of size 2^j that start at index_start and end at index_end, respectively
        if self.table[index_start][j] > self.table[index_end - (1 << j) + 1][j]:
            return self.table[index_start][j]
        else:
            return self.table[index_end - (1 << j) + 1][j]
