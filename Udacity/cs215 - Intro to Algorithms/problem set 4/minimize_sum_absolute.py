#
# Given a list of numbers, L, find a number, x, that
# minimizes the sum of the absolute value of the difference
# between each element in L and x: SUM_{i=0}^{n-1} |L[i] - x|
#
# Your code should run in Theta(n) time
#

import random as rand
from math import ceil


# Write partition to return a new array with
# all values less then `v` to the left
# and all values greater then `v` to the right
def partition(L, v):
    P = []
    higher_list = []
    num_v = 0
    for item in L:
        if v > item:
           P.append(item)  # adds item to lesser side of new list (unordered)
        elif v < item:
            higher_list.append(item)  # add to higher side list
        elif v == item:
            num_v += 1
    # what if the item is equal to v? I have just added them all to middle of two partitions...
    for _ in range(num_v):
        P.append(v)
    P += higher_list
    return P


# given this function - I have commented on it
def top_k(L, k):
    # a recursive algorithm to find the top/bottom k values (unsorted) in a list using randomly chosen pivots
    v = L[rand.randrange(len(L))]
    # TODO: does not handle doubled values, must look to change here and partition
    (left, middle, right) = partition(L, v)
    # if lower list = k, then we have found all k values, with v being 1 above that
    if len(left) == k:
        return left
    # if lower list + 1 = k then, the lower list + v constitutes all k values
    if len(left)+1 == k:
        return left+[v]
    # if lower list is larger than k, then we have too many values, and recursively call this function
    # still using value k to partition the lower list to make it smaller, till it reaches length k
    if len(left) > k:
        return top_k(left, k)
    # if lower list is smaller than k, then we have too few values, and recursively call the function
    # using a k value lowered by the length of the left list to start a new partition on the right list
    # till the length of the new lower list added to length of original lower list = k
    return left+[v]+top_k(right, k-len(left)-1)


def minimize_absolute(L):
    # this is same as finding the median in the sorted list, which we can get by partitioning to
    # find lowest k values, where k = list size/2, essentially splitting list in half,
    # then we search the list for the highest value which will be the median
    # this takes n operations for the top_k part and 1/2n operations to find max in the return list

    # if list is even, will take the lower of two middle values, if odd, ceil gives middle value
    length_list = len(L)
    half_vals = ceil(length_list / float(2))  # used float as udacity seems to not cast this to a float automatically
    lower_half = top_k(L, half_vals)
    median = lower_half[0]
    # get highest value in list, which is the median
    for val in lower_half:
        if val > median:
            median = val
    return median
