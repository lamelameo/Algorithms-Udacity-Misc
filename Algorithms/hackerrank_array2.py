""" Hackerrank array problem: https://www.hackerrank.com/challenges/array-pairs/problem
Given an array of n integers, A = [a_1, a_2, a_3, ..., a_n]. Find the total number of (i,j) pairs such that
a_i x a_j <= max(a_i, a_i+1, ..., a_j), where i < j.
Constraints:
    1 <= n <= 10^5
    1 <= a_i <= 10^9
"""
import random
import itertools
import math
import time


# Simple solution, scan all pairs sequentially, building up max as we go. Theta(n^2) time
# TODO: is there any method to reduce scan below all n^2 pairs for all inputs?
def simple_pairs(array):
    count = 0
    length = len(array)
    for i in range(length):
        a_i = array[i]
        maximum = a_i
        # print(i)
        for j in range(i + 1, length):
            # print('test')
            a_j = array[j]
            if a_j > maximum:
                maximum = a_j
            if maximum >= (a_i * a_j):
                count += 1
    return count


# array = [random.randint(1, 1000000000) for _ in range(10000)]
# TODO: list of alternating low and high numbers, should be ~(n^2/4) good pairs as all low num pairs are good
# array = []
# for _ in range(5000):
#     array.append(random.randint(1000000, 1000000000))
#     array.append(random.randint(1, 30000))
# print("\npairs: ", simple_pairs(array))
# quit()
# array = [2,48,24,12,6,36,18,3]
# array = [4,2,6000,1,100,898818,9939999,9999129,5939889]
# results = []
# for permutation in itertools.permutations(array, len(array)):
#     results.append((permutation, simple_pairs(permutation)))
#     print("\npairs: ", simple_pairs(array))
# results.sort(key=lambda key: key[1])
# print(results[-1])
# quit()


# TODO: can have half good half bad pairs even with unique ints but need exponentially increasing integers as n grows...
#  max=~200,000 for n=8. If we try minimize growth of integers by adding smallest a_i*a_j each step, then bad pairs
# There are theta(n^2) pairs, but the max and probable satisfiable pairs are much lower given that our integers are only
# max 10^9 with array size max 10^5, can just do an array sorting (assuming unique or uniformly distributed integers)
# taking nlogn, then scan pairs starting at lowest ints, calculating their multiplication and comparing to max in array,
# then downwards to lowest num? binary scan? or just check the max of all nums within both
# TODO: max heap to find max faster? need to find max in certain range fast -> segment tree = O(1) max, O(nlogn) space
# at some point all pairs will just go over max immediately, therefore can terminate here
# If we have repetitions of numbers, or 1's then we devolve to theta(n^2)
# CAN WE???
# we can avoid this by skipping repeats and changing the first repeat to a negative of the 1st index of non repeat,
# then anytime we hit a repeat we simply skip to that index, avoiding unnecessary work...

# Overall we have nlogn + n^alpha where alpha < 2?? Idk the exact max on possible satisfiable pairs
def pairs(array):
    # iterate from lowest integers forming multiplicative pairs, test iterating backwards from highest integers
    # lower bound constant time but devolves to O(n^2)*T(range max) for pathological input such as all 1's?
    ary_sorted = [x for x in enumerate(array)]
    ary_sorted.sort(key=lambda key: key[1])
    # TODO: cutoff for pair multiplications to be considered is not max in list, its max / min?
    #  can keep updating val as we move up minimum vals, so max goes down as min goes up
    ary_max = ary_sorted[-1][1]
    print("arrays: ", ary_sorted[:50])
    length = len(ary_sorted)
    count = 0
    loops = 0
    # iterate through i values
    flag1 = 0
    for i in range(length):
        # print("\ni: ", i)
        flag2 = 0
        if flag1:
            break
        # iterate through remaining values to get j to form the pair
        for j in range(i + 1, length):
            # print("j: ", j)
            if flag2:
                break
            # value of each item of current pair
            pair_i = ary_sorted[i][1]
            pair_j = ary_sorted[j][1]
            # index of each item of current pair
            low_pair = min(ary_sorted[i][0], ary_sorted[j][0])
            high_pair = max(ary_sorted[i][0], ary_sorted[j][0])
            # in case either integer = 1, don't bother searching
            if pair_i == 1 or pair_j == 1:
                # print("pair with 1 found")
                count += 1
                continue
            # iterate backwards from highest value to see if we can satisfy the condition for the pair
            for comp in range(-1, -(length - j), -1):
                loops += 1
                # print("compare: ", comp)
                # pair multiplication is greater than current comparison value
                if (pair_i * pair_j) > ary_sorted[comp][1]:
                    # abort scan once our 1st pair item has no 2nd pair items satisfiable, as no further pairs will
                    # be satisfiable as they are all even larger than current item
                    if (j == i+1) and (comp == -1):
                        print("stopping: ", pair_i, pair_j)
                        flag1 = 1
                        flag2 = 1
                    break
                # check if comp is in range of pair
                elif low_pair < ary_sorted[comp][0] < high_pair:
                    # print("pair found: {} x {} <= {} ".format(pair_i, pair_j, ary_sorted[comp][1]))
                    # print("range: {} ... {} ... {}".format(low_pair, ary_sorted[comp][0], high_pair))
                    count += 1
                    break
    # TODO: running time degrades to theta(n^3) if we have bad inputs such as alternating low and high ints,
    #  e.g. 1,x,1,x,...,1 ~1/4 pairs need check = O(n^3) or all 1's = O(n^2) (handled in O(1))
    #  can get theta(n^2) if we use a segment tree for O(1) max query
    print("loops: ", loops)
    return count


def pairs_table(array):
    # iterate from lowest integers forming multiplicative pairs, test iterating backwards from highest integers
    # lower bound constant time but devolves to O(n^2)*T(range max) for pathological input such as all 1's?
    ary_sorted = [x for x in enumerate(array)]
    ary_sorted.sort(key=lambda key: key[1])
    ary_max = ary_sorted[-1][1]
    print("arrays: ", ary_sorted[:50])
    length = len(ary_sorted)
    count = 0
    loops = 0
    timer2 = time.clock()
    sparse_table = SparseTable(array)
    print("table construction took: ", time.clock() - timer2)
    # for item in sparse_table.table:
    #     print(item)
    # iterate through i values
    flag1 = 0
    for i in range(length):
        if i % 1000 == 0:
            print('test')
        if flag1:
            break
        # iterate through remaining values to get j to form the pair
        for j in range(i + 1, length):
            loops += 1
            # value of each item of current pair
            pair_i = ary_sorted[i][1]
            pair_j = ary_sorted[j][1]
            # index of each item of current pair
            low_pair = min(ary_sorted[i][0], ary_sorted[j][0])
            high_pair = max(ary_sorted[i][0], ary_sorted[j][0])
            # in case either integer = 1, don't bother searching
            if pair_i == 1 or pair_j == 1:
                count += 1
                continue
            # pair multiplication is greater than current comparison value
            if (pair_i * pair_j) > ary_max:
                # abort scan once our 1st pair item has no 2nd pair items satisfiable, as no further pairs will
                # be satisfiable as they are all even larger than current item
                if j == i + 1:
                    # print("stopping: ", pair_i, pair_j)
                    flag1 = 1
                    break
            # check if comp is in range of pair
            elif (pair_i * pair_j) <= sparse_table.max(low_pair, high_pair):
                count += 1
    print("loops: ", loops)
    return count


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
            for i in range(n - 2**j + 1):
                # max for this size 2^j range is max of two sized 2^(j-1) ranges calculated in previous loop
                self.table[i][j] = max(self.table[i][j - 1], self.table[i + 2**(j - 1)][j - 1])

    def max(self, index_start, index_end):
        # get biggest power of 2 <= size of given range
        j = int(math.log2(index_end - index_start + 1))
        # calculate maximum based on the two table items of that size that cover the range
        # ie the 2 items of size 2^j that start at index_start and end at index_end, respectively
        return max(self.table[index_start][j], self.table[index_end - 2**j + 1][j])


# print("\npairs: ", pairs([1, 2, 6, 3, 4]))
ary = []
for _ in range(5000):
    ary.append(random.randint(1000000, 1000000000))
    ary.append(random.randint(1, 30000))
# ary = [random.randint(1, 1000000000) for _ in range(100000)]
# ary = [2, 48, 24, 12, 6, 36, 18, 3]
timer = time.clock()
print("\npairs: ", simple_pairs(ary))
print("scan took: ", time.clock() - timer)
print()
timer = time.clock()
print("\npairs: ", pairs_table(ary))
print("table took: ", time.clock() - timer)
# table = SparseTable([7, 2, 3, 0, 5, 10, 3, 12, 18, 19])
# for item in table.table:
#     print(item)
# print()
# print(table.table[2])
# print(table.max(7, 7))


def cuda():
    import numpy as np
    import time
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    ary = np.ones(100000, np.uintc)
    count = np.array([0], np.uintc)
    kernel = SourceModule("""
    #include <stdio.h>
    __global__ void scan(int *input, int *counter)
    {
        int index = threadIdx.x + (blockIdx.x * 256);
        if(index > 100000) {
            return;
        }
        int end = 100000 - index;
        for(int i=0; i<end; i++) {
            atomicAdd(counter, input[index]);
        }
    }
    """)
    func = kernel.get_function("scan")
    ary_gpu = gpuarray.to_gpu_async(ary)
    counter_gpu = gpuarray.to_gpu_async(count)
    timer1 = time.clock()
    func(ary_gpu, counter_gpu, block=(256, 1, 1), grid=(400, 1))
    print(time.clock() - timer1)
    print(counter_gpu.get())
