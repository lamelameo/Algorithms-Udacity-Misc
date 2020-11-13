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
from collections import deque
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
    for _ in range(10000):
        ary.append(random.randint(1000000, 1000000000))
        ary.append(random.randint(1, 30000))
    # for _ in range(5000):
        # ary.append(random.randint(1000000, 1000000000))
        # ary.append(random.randint(1, 30000))
    # ary = [random.randint(1, 1000000000) for _ in range(100000)]

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
    # timer = time.clock()
    # print("\npairs:", simple_pairs(ary))
    # print("time:", time.clock() - timer)
    timer = time.clock()
    print("\npairs iterated tree:", iterated_tree(ary))
    print("time:", time.clock() - timer)
    print()
    timer = time.clock()
    print("\npairs iterated:", iterated(ary))
    print("time:", time.clock() - timer)
    print()
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
count, counter1, counter2 = 0, 0, 0
def recursive(array, start, end):
    if not array:
        return 0
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

        # recurse on left and right sub arrays if they exist, recursing on the smaller first
        if left > right:
            pairs += recursive(array, start, max_ind-1)
            pairs += recursive(array, max_ind+1, end)
        else:
            pairs += recursive(array, max_ind + 1, end)
            pairs += recursive(array, start, max_ind - 1)
        # print("max:", max_x, "pairs:", pairs)
        return pairs

    # sub array of size 1 or 0 has no further pairs
    else:
        count += 1
        return 0


# iterated version of recursive algorithm, so we dont get stack overflow when we get bad case arrangement of maximums
def iterated(array):
    pairs = 0
    iterations = deque()
    iterations.append((0, len(array) - 1))
    # we will need to loop over all items only once in order dictated by recursive algo
    while iterations:
        start, end = iterations.pop()
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
            iterations.append((start, max_ind - 1))
        if right:
            iterations.append((max_ind + 1, end))
    return pairs


# TODO: can we do a sweep first to find where at the maxes are and then create appropriately sized dicts?
#   OR: create and destroy both left and right trees each "recursion" rather than storing them in array/dict?
#      except this is O(n^2logn) for the case we are tryign to improve
# iterated algorithm using trees instead of array, makes the worst case of iteration using arrays better
# ie when array is in reverse sorted order or close to this.
# this algorithm is faster in this case, but 10x slower for average orderings of maximums where we get logn recursions
def iterated_tree(array):
    # TODO: need to keep index in unsorted array in tree nodes, this is so we can recurse once we find max
    #  2d tree with index as 2nd axis? no size attribute - use index for order?
    #  we cant discriminate position of duplicate values currently
    #  Use dict to map item to index instead?
    pairs = 0
    tree_num = 0
    # index of each item in unsorted array is mapped to its value in dict, handling duplicates with a list
    indexes = dict()
    for index, item in enumerate(array):
        entry = indexes.get(item)
        if entry:
            entry.append(index)
        else:
            indexes[item] = [index]
    iterations = deque()
    trees = [AVLTree(array)]
    iterations.append((0, 0, len(array) - 1))
    # we will need to loop over all items only once in order dictated by recursive algo
    while iterations:
        # print("queue:", iterations)
        # print("trees:", [tree.tree for tree in trees])
        tree_ind, start, end = iterations.pop()
        # get the max of the sub array then split into left and right trees
        tree = trees[tree_ind]
        # print("\ntree:", tree.tree.keys())
        max_x = tree.max()
        tree.remove(max_x)
        # get the max index in un
        # sorted array, must check the list if there are duplicates
        max_ind = 0
        maxes = indexes[max_x]
        for x in maxes:
            if start <= x <= end:
                max_ind = x
                break
        tree2 = AVLTree()
        # print("max:", max_x)
        # construct right and left by scanning smaller side and removing these values and placing into new tree
        if (end - max_ind) <= (max_ind - start):
            flag = True
            # max is closer to end than start, tree = left, tree2 = right
            for x in array[max_ind+1: end+1]:
                # print("x:", x)
                tree.remove(x)
                tree2.insert(x)
        else:
            flag = False
            # tree = right, tree2 = left
            for x in array[start: max_ind]:
                tree.remove(x)
                tree2.insert(x)
        # print("tree:", tree.tree.keys(), "\ntree2:", tree2.tree.keys())
        # print(tree.tree)
        # TODO: bug is due to children attribute not being updated properly
        # in either case tree2 is smaller, so scan it and find number of pairs
        for i in tree2.tree:
            mult = int(max_x / i)
            # print("i:", i, "max/i:", mult)
            # print("pairs:", tree.rank(mult))
            # must take into account duplicates as they are stored in one tree item
            pairs += tree.rank(mult) * tree2.tree.get(i)[6]
        # handle pairs of the sort; (1, max) or (max, 1) and (max=1, i) or (i, max=1)
        if max_x == 1:
            pairs += tree.size + tree2.size
        else:  #
            if tree.size > 0:
                pairs += tree.rank(1)
            if tree2.size > 0:
                pairs += tree2.rank(1)
        # add left and right trees to recursion queue if they exist, add small tree last, so we recurse on them first
        # to limit queue size

        # flag = True if tree = left
        # larger tree is non zero
        if tree.size > 0:
            # is left tree
            if flag:
                # print('left big:', tree.tree)
                iterations.append((tree_ind, start, max_ind - 1))
            # is right tree
            else:
                # print('right big:', tree.tree)
                iterations.append((tree_ind, max_ind + 1, end))
        # smaller tree is non zero
        if tree2.size > 0:
            tree_num += 1
            trees.append(tree2)
            # is right tree
            if flag:
                # print('right small:', tree2.tree)
                iterations.append((tree_num, max_ind + 1, end))
            # is left tree
            else:
                # print('left small:', tree2.tree)
                iterations.append((tree_num, start, max_ind - 1))

    return pairs


def test_2():
    array = [1]
    # # for _ in range(10):
    # #     array.append(random.randint(2, 1000))
    # for _ in range(10):
    #     array.append(random.randint(1, 100))
    # # array.append(1)
    # array = [2, 2, 61, 21, 5, 58, 78, 13, 20, 26]
    # print(iterated_tree(array))
    # print(simple_pairs(array))
    #
    # flag = True
    # x, y = 1, 1
    # while x == y:
    #     array = []
    #     for _ in range(10):
    #         array.append(random.randint(1, 100))
    #     x = iterated_tree(array)
    #     y = simple_pairs(array)
    #
    # print(array)
    file = open("..\\..\\learning\\array2testdata.txt")
    string = ""
    array = []
    for line in file:
        string += line
    file.close()
    num = ""
    for letter in string:
        if letter == " ":
            array.append(int(num))
            num = ""
        else:
            num += letter

    timer = time.clock()
    print("\npairs iterated tree:", iterated_tree(array))
    print("time:", time.clock() - timer)
    quit()


test_2()


# def recursive_tree(tree):
#     size_ = end - start + 1
#     # find max, determine number of pairs and recurse on sub arrays on either side of the max, then add
#     if size_ > 1:
#         pairs = 0
#         max_x = 0
#         max_ind = 0
#         # get max then split into left and right sub arrays
#         for index in range(start, end + 1):
#             count += 1
#             if array[index] > max_x:
#                 max_x = array[index]
#                 max_ind = index
#         # determine number of pairs using left and right sub arrays
#         count += size_ - 1
#         count += 2 * size_ * math.log2(size_ + 1)
#         left = [x for x in array[start: max_ind]]
#         right = [x for x in array[max_ind + 1: end + 1]]
#         # print("l,r", left, right)
#         left.sort()
#         right.sort()
#         count += len(left) * math.log2(len(right) + 1)
#
#         # scan smaller sub array of left or right and binary search other array for pairs
#         (scan, match) = (left, right) if left <= right else (right, left)
#         for i in scan:
#             mult = max_x / i
#             pairs += binary_search(match, mult)
#         # handle pairs of the sort; (1, max) or (max, 1) and (max=1, i) or (i, max=1)
#         if max_x == 1:
#             # print('test', left, right)
#             pairs += len(right)
#             pairs += len(left)
#         else:  # if left or right is None then binary search will return 0
#             count += math.log2(len(left) + 1) + math.log2(len(right) + 1)
#             pairs += binary_search(left, 1)
#             pairs += binary_search(right, 1)
#
#         # recurse on left and right sub arrays if they exist
#         if left:
#             pairs += recursive(array, start, max_ind - 1)
#         if right:
#             pairs += recursive(array, max_ind + 1, end)
#         # print("max:", max_x, "pairs:", pairs)
#         return pairs
#
#     # sub array of size 1 or 0 has no further pairs
#     else:
#         count += 1
#         return 0


test()
# print("simple:", simple_pairs([5,1,1,10,1,5,1,3]))
# print("recursive:", recursive([5,1,1,10,1,5,1,3], 0, 7))
# TODO: recursion depth limit
# print('simple', simple_pairs([x for x in range(990, 0, -1)]))
# print('test', recursive([x for x in range(990, 0, -1)], 0, 989))
# print('test', recursive([random.randint(1, 1000000000) for _ in range(100000)], 0, 99999))
print("time:", count, "n * 2(logn)^2", (2 * 10**5) * (math.log2(10**5))**2)
print("left scan:", counter2, "sqrt scan:", counter1)
