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
import queue
from AVLTree import AVLTree
from SparseTable import SparseTable


# Simple solution, scan all pairs sequentially, building up max as we go. Theta(n^2) time
# TODO: is there any method to reduce scan below all n^2 pairs for all inputs?
#  can we augment this algo to be faster?
def simple_pairs(array):
    count = 0
    length = len(array)
    loops = 0
    for i in range(length):
        a_i = array[i]
        # print()
        # print("a_i", a_i)
        maximum = a_i
        # print(i)
        for j in range(i + 1, length):
            # print('test')
            loops += 1
            a_j = array[j]
            # print("a_j", a_j)
            if a_j > maximum:
                maximum = a_j
            if maximum >= (a_i * a_j):
                # print('+1')
                count += 1
    print("loops: ", loops)
    return count


# TODO: can have half good half bad pairs even with unique ints but need exponentially increasing integers as n grows...
#  max=~200,000 for n=8. If we try minimize growth of integers by adding smallest a_i*a_j each step, then bad pairs
#  grow much slower, looks like about O(sqrt(n)) but havent calculated actual number
# There are theta(n^2) pairs, but the max and probable satisfiable pairs are much lower given that our integers are only
# max 10^9 with array size max 10^5, can just do an array sorting (assuming unique or uniformly distributed integers)
# taking nlogn, then scan pairs starting at lowest ints, calculating their multiplication and comparing to max in array,
# then downwards to lowest num? binary scan? or just check the max of all nums within both
# TODO: max heap to find max faster? need to find max in certain range fast -> segment tree = O(1) max, O(nlogn) space
# at some point all pairs will just go over max immediately, therefore can terminate here
# If we have repetitions of numbers, or 1's then we devolve to theta(n^2)

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
        pair_i = ary_sorted[i][1]
        # iterate through remaining values to get j to form the pair
        for j in range(i + 1, length):
            # print("j: ", j)
            if flag2:
                break
            # value of each item of current pair
            pair_j = ary_sorted[j][1]
            # index of each item of current pair
            if ary_sorted[i][0] < ary_sorted[j][0]:
                low_pair, high_pair = ary_sorted[i][0], ary_sorted[j][0]
            else:
                low_pair, high_pair = ary_sorted[j][0], ary_sorted[i][0]
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
    #  e.g. 1,x,1,x,...,1 ~3/4 pairs need check = O(n^3) or all 1's = O(n^2) (handled in O(1))
    #  can get theta(n^2) if we use a sparse array for O(1) max query
    print("loops: ", loops)
    return count


def pairs_table(array):
    # iterate from lowest integers forming multiplicative pairs, test iterating backwards from highest integers
    # lower bound constant time but devolves to O(n^2)*T(range max) for pathological input such as all 1's?
    ary_sorted = [x for x in enumerate(array)]
    ary_sorted.sort(key=lambda key: key[1])
    ary_max = ary_sorted[-1][1]
    length = len(ary_sorted)
    count = 0
    loops = 0
    sparse_table = SparseTable(array)
    for i in range(length):
        pair_i = ary_sorted[i][1]
        # iterate through remaining values to get j to form the pair
        for j in range(i + 1, length):
            loops += 1
            # value of each item of current pair
            pair_j = ary_sorted[j][1]
            # index of each item of current pair
            if ary_sorted[i][0] < ary_sorted[j][0]:
                pair_start, pair_end = ary_sorted[i][0], ary_sorted[j][0]
            else:
                pair_start, pair_end = ary_sorted[j][0], ary_sorted[i][0]
            # # in case either integer = 1, don't bother searching
            # if pair_i == 1 or pair_j == 1:
            #     count += 1
            #     continue
            # pair multiplication is greater than current comparison value
            if pair_i * pair_j > ary_max:
                # abort scan once our 1st pair item has no 2nd pair items satisfiable, as no further pairs will
                # be satisfiable as they are all even larger than current item
                if j == i + 1:
                    print("loops: ", loops)
                    return count
            # check if comp is in range of pair
            elif pair_i * pair_j <= sparse_table.max(pair_start, pair_end):
                count += 1

    print("loops: ", loops)
    return count


def test():
    # TODO: list of alternating low and high numbers, should be ~O((3/4)n^2) pairs to check as all pairs with both
    #  high numbers are ones we can skip and there are only O(n^2/4) of them

    # TODO: check number of items in list <= sqrt max then run appropriate algo
    ary = [1]
    for _ in range(5000):
        ary.append(random.randint(1000000, 1000000000))
        # ary.append(random.randint(1, 30000))
    for _ in range(5000):
        # ary.append(random.randint(1000000, 1000000000))
        ary.append(random.randint(1, 30000))
    # ary = [random.randint(1, 1000000000) for _ in range(10000)]

    # TODO: input array which guarantees max value is in middle of each sub array at all depths
    # ary.sort()
    rec = [x for x in range(1, 10001)]
    rec_ary = [0 for _ in range(10000)]

    def recurse(start, end):
        # mid = int((start + end) / 2)
        # rec_ary[mid] = ary.pop()
        # if mid + 1 <= end:
        #     recurse(mid + 1, end)
        # if start <= mid - 1:
        #     recurse(start, mid - 1)

        num = end - start
        if num:
            mid = start + int(math.sqrt(end - start))
            # print(mid)
            rec_ary[mid] = ary.pop()
            # print(rec_ary)
        else:
            rec_ary[start] = rec.pop()
            return
        if start <= mid - 1:
            # print("left")
            recurse(start, mid - 1)
        if mid + 1 <= end:
            # print("right")
            recurse(mid + 1, end)

    # recurse(0, 9999)
    timer = time.clock()
    print("\npairs:", simple_pairs(ary))
    print("time:", time.clock() - timer)
    timer = time.clock()
    print("\npairs iterated:", iterated(ary))
    print("time:", time.clock() - timer)
    timer = time.clock()
    print("\npairs recursive:", recursive(ary, 0, len(ary) - 1))
    print("time:", time.clock() - timer)
    print()
    # print("time:", count, "n * 2(logn)^2", (2 * 10 ** 5) * (math.log2(10 ** 5)) ** 2)
    # print("left scan:", counter2, "sqrt scan:", counter1)
    timer = time.clock()
    print("pairs table:", pairs_table(ary))
    print("time:", time.clock() - timer)
    quit()

    # TODO: testing inserting lowest i,j pairs each step in array at worst spot for running time...make sure it is worst
    ary = [2, 48, 24, 12, 6, 36, 18, 3]
    # ary = [4,2,6000,1,100,898818,9939999,9999129,5939889]
    results = []
    for permutation in itertools.permutations(ary, len(ary)):
        results.append((permutation, simple_pairs(permutation)))
        # print("\npairs: ", simple_pairs(ary))
    results.sort(key=lambda key: key[1])
    print(results[-1])


# find the index where an item would be placed in a sorted array
def binary_search(ary, item):
    first = 0
    last = len(ary) - 1
    # loop till first < last, then take first as our index, as in all cases this is right choice
    # also should handle duplicates, returning index after last duplicate
    while first <= last:
        mid = (last + first) // 2  # take floor of floats
        if item < ary[mid]:  # less than mid value
            last = mid - 1
        else:  # greater than or equal to mid value
            first = mid + 1
    return first


# scan all n values and use them as potential maximum values, searching either side of current value for i,j pairs that
# satisfy the condition. Keep each side sorted then we can some magic to find num pairs satisfied
# Using arrays means we have O(n) each step because insert/delete may be at start of array but testing speed anyway
def array_scan(array):
    count = 0
    n = len(array)
    # list of left arrays, corresponding max x used, and max R value paired
    left = [(array[1], None, [array[0]])]
    # right array
    right = [x for x in array[1:]]
    right.sort()

    # add pairs with 1 and next index item, as they arent covered with algo
    count += binary_search(right, 1)
    # if last item is 1, it has no next item to pair with
    if array[-1] == 1:
        count -= 1

    for x in array:
        # find where x is in right array and remove it
        x_ = binary_search(right, x) - 1
        # TODO: do magic here

        # find where x should be placed in left array and insert it
        left.insert(binary_search(left, x), x)

    return count


# TODO: do same scan on n items as potential maximums for pairs, but use AVL tree instead of sorted lists
# build = nlogn, can do insert, delete, and search in Θ(logn) so our overall time will be Θ(nlogn)?
def avl_scan(array):
    # can store the max for each num in L and R, then we dont have to store multiple trees, as we just search for pairs
    # for all nums below sqrt x in L,R
    left_max = dict()
    right_max = dict()
    for item in array:
        left_max[item] = 0
        right_max[item] = 0
    left = AVLTree([array[0]])
    right = AVLTree(array[1:])

    # TODO: if x is min value in L or R, will we still find pairs with 1?
    #  also if item has size > 1 then have to take this into account


# O(n(logn)^2) solution (with AVL tree, O(n^2) with array?)
count = 0
counter1 = 0
counter2 = 0
def recursive(array, start, end):
    global count, counter1, counter2
    size_ = end - start + 1
    # find max, determine number of pairs and recurse on sub arrays on either side of the max, then add
    if size_ > 1:
        pairs = 0
        max_x = 0
        max_ind = 0
        # get max then split into left and right sub arrays
        for index in range(start, end + 1):
            count += 1
            if array[index] > max_x:
                max_x = array[index]
                max_ind = index
        # determine number of pairs using left and right sub arrays
        count += size_ - 1
        count += 2 * size_ * math.log2(size_ + 1)
        left = [x for x in array[start: max_ind]]
        right = [x for x in array[max_ind + 1: end+1]]
        # print("l,r", left, right)
        left.sort()
        right.sort()
        count += len(left) * math.log2(len(right) + 1)
        # TODO: just use all left items to find pairs, could use sqrt max in L,R instead
        #  is only faster in worst case if we use a tree, as we can hold duplicates in one item
        #  even with array we have a better best case though?
        # root = int(math.sqrt(max_x))
        # left_end = binary_search(left, root)
        # right_end = binary_search(right, root)
        # counter1 += left_end * math.log2(len(right) + 1)
        # counter1 += right_end * math.log2(len(left) + 1)
        # counter2 += len(left) * math.log2(len(right) + 1)
        # count += left_end * math.log2(len(right) + 1)
        # count += right_end * math.log2(len(left) + 1)
        # for i in range(left_end):
        #     mult = max_x/left[i]
        #     pairs += binary_search(right, mult)
        # for j in range(right_end):
        #     mult = max_x/right[j]
        #     pairs += binary_search(left, mult)

        # scan smaller sub array of left or right and binary search other array for pairs
        (scan, match) = (left, right) if left <= right else (right, left)
        for i in scan:
            mult = max_x / i
            pairs += binary_search(match, mult)
        # handle pairs of the sort; (1, max) or (max, 1) and (max=1, i) or (i, max=1)
        if max_x == 1:
            # print('test', left, right)
            pairs += len(right)
            pairs += len(left)
        else:  # if left or right is None then binary search will return 0
            count += math.log2(len(left) + 1) + math.log2(len(right) + 1)
            pairs += binary_search(left, 1)
            pairs += binary_search(right, 1)

        # recurse on left and right sub arrays if they exist
        if left:
            pairs += recursive(array, start, max_ind-1)
        if right:
            pairs += recursive(array, max_ind+1, end)
        # print("max:", max_x, "pairs:", pairs)
        return pairs

    # sub array of size 1 or 0 has no further pairs
    else:
        count += 1
        return 0


def iterated(array):
    pairs = 0
    iterations = queue.Queue()
    iterations.put((0, len(array) - 1))
    # we will need to loop over all items only once in order dictated by recursive algo
    while iterations.qsize():
        start, end = iterations.get_nowait()
        max_x = 0
        max_ind = 0
        # find the max of the sub array then split into left and right sections
        for index, item in enumerate(array[start:end+1]):
            if item > max_x:
                max_x = item
                max_ind = start + index
        left = array[start:max_ind]
        right = array[max_ind+1:end+1]
        left.sort()
        right.sort()
        # calculate num of pairs, scanning smaller sub array of left or right
        (scan, match) = (left, right) if left <= right else (right, left)
        for i in scan:
            mult = max_x / i
            pairs += binary_search(match, mult)
        # handle pairs of the sort; (1, max) or (max, 1) and (max=1, i) or (i, max=1)
        if max_x == 1:
            pairs += len(right)
            pairs += len(left)
        else:  # if left or right is None then binary search will return 0
            pairs += binary_search(left, 1)
            pairs += binary_search(right, 1)
        # add left and right sub arrays to recursion queue if they exist
        if left:
            iterations.put((start, max_ind - 1))
        if right:
            iterations.put((max_ind + 1, end))
    return pairs


test()
# print("simple:", simple_pairs([5,1,1,10,1,5,1,3]))
# print("recursive:", recursive([5,1,1,10,1,5,1,3], 0, 7))
# TODO: recursion depth limit
# print('simple', simple_pairs([x for x in range(990, 0, -1)]))
# print('test', recursive([x for x in range(990, 0, -1)], 0, 989))
# print('test', recursive([random.randint(1, 1000000000) for _ in range(100000)], 0, 99999))
print("time:", count, "n * 2(logn)^2", (2 * 10**5) * (math.log2(10**5))**2)
print("left scan:", counter2, "sqrt scan:", counter1)
