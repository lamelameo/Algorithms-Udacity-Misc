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
    loops = 0
    for i in range(length):
        a_i = array[i]
        maximum = a_i
        # print(i)
        for j in range(i + 1, length):
            # print('test')
            loops += 1
            a_j = array[j]
            if a_j > maximum:
                maximum = a_j
            if maximum >= (a_i * a_j):
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
    print("arrays: ", ary_sorted[:50])
    length = len(ary_sorted)
    count = 0
    loops = 0
    timer2 = time.clock()
    # sparse_table = SparseTable(array)
    table = [[0 for _ in range(1 + int(math.log2(len(array))))] for _ in range(len(array))]
    n = len(array)
    # run through size 1 range maxes first so we can use 2^(j-1) maxes each remaining loop
    for i in range(n):
        table[i][0] = array[i]
    # loop through ranges sized 2^j for j = 1 to floor(logn)
    for j in range(1, 1 + int(math.log2(n))):
        # calc max for size 2^j ranges (there are n - 2^j of them)
        for i in range(n - (1 << j) + 1):
            # max for this size 2^j range is max of two sized 2^(j-1) ranges calculated in previous loop
            table[i][j] = max(table[i][j - 1], table[i + (1 << (j - 1))][j - 1])
    print("table construction took: ", time.clock() - timer2)
    # for item in sparse_table.table:
    #     print(item)
    # iterate through i values
    for i in range(length):
        # if i % 1000 == 0:
        #     print('test')
        pair_i = ary_sorted[i][1]
        # iterate through remaining values to get j to form the pair
        for j in range(i + 1, length):
            loops += 1
            # value of each item of current pair
            pair_j = ary_sorted[j][1]
            # index of each item of current pair
            # pair_start = min(ary_sorted[i][0], ary_sorted[j][0])
            # pair_end = max(ary_sorted[i][0], ary_sorted[j][0])
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
                    # print("stopping: ", pair_i, pair_j)
                    print("loops: ", loops)
                    return count
            # check if comp is in range of pair
            else:
                # get biggest power of 2 <= size of given range
                j = int(math.log2(pair_end - pair_start + 1))
                # calculate maximum based on the two table items of that size that cover the range
                # ie the 2 items of size 2^j that start at index_start and end at index_end, respectively
                if table[pair_start][j] > table[pair_end - (1 << j) + 1][j]:
                    range_max = table[pair_start][j]
                else:
                    range_max = table[pair_end - (1 << j) + 1][j]
                if pair_i * pair_j <= range_max:
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
        if self.table[index_start][j] > self.table[index_end - 2**j + 1][j]:
            return self.table[index_start][j]
        else:
            return self.table[index_end - 2**j + 1][j]


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

    timer = time.clock()
    print("\npairs: ", simple_pairs(ary))
    print("scan took: ", time.clock() - timer)
    print()
    timer = time.clock()
    print("\npairs: ", pairs_table(ary))
    print("table took: ", time.clock() - timer)

    # TODO: testing inserting lowest i,j pairs each step in array at worst spot for running time...make sure it is worst
    ary = [2, 48, 24, 12, 6, 36, 18, 3]
    # ary = [4,2,6000,1,100,898818,9939999,9999129,5939889]
    results = []
    for permutation in itertools.permutations(ary, len(ary)):
        results.append((permutation, simple_pairs(permutation)))
        # print("\npairs: ", simple_pairs(ary))
    results.sort(key=lambda key: key[1])
    print(results[-1])

    # table = SparseTable([7, 2, 3, 0, 5, 10, 3, 12, 18, 19])
    # for item in table.table:
    #     print(item)
    # print()
    # print(table.table[2])
    # print(table.max(7, 7))


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

    # find the index where an item would be placed in a sorted array
    def binary_search(ary, item):
        first = 0
        last = len(ary) - 1
        # loop till first < last, then take first as our index, as in all cases this is right choice
        # also should handle duplicates, returning index after last duplicate
        while first <= last:
            mid = (last + first) // 2  # take floor of floats
            if item < array[mid]:  # less than mid value
                last = mid - 1
            else:  # greater than or equal to mid value
                first = mid + 1
        return first

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
# build = nlogn, can do insert, delete, and search in Θ(logn) so our overall time will be Θ(nlogn)
# ie build + scan*(4*range max + search + insert + delete)
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


class AVLTree:
    """ AVL tree data structure supporting O(logn) search, insert, and delete queries in O(n) space
    Takes one optional parameter: array - a List object which will be converted to a tree
    Each node in the tree is a list containing 7 items: (key, parent, left, right, balance, children, size)
    Using the children attribute we can determine the rank of item in sorted list
    The size attribute stores the number of this key that are present in the set
     """
    def __init__(self, array=None):
        # TODO: use dict for pointer like functionality? This also means we have O(1) search not O(logn)??
        #  are there losses for dict for add/del - preprocess dict with array item with empty tuples?
        # self.tree = []
        self.tree = dict()
        self.root = None
        # we have been given an array to convert to a tree
        if array:
            self.root = array[0]
            self.tree[array[0]] = [array[0], None, None, None, 0, 0, 1]
            for item in array[1:]:
                self.insert(item)

    # find the parent node to which the item should be attached, or the item's node if it already exists
    # return a tuple, tuple[0] is the node key and tuple[1] is 1 or 0 representing item node or parent node
    def search(self, item):
        node = self.tree.get(item)
        if node:
            return node, 1
        else:
            current = self.tree[self.root]
            # traverse tree
            while True:
                if item < current[0]:  # item less than node - go left
                    if current[2]:
                        current = self.tree[current[2]]
                    else:
                        return current, 0
                else:  # go right
                    if current[3]:
                        current = self.tree[current[3]]
                    else:
                        return current, 0

    # TODO: insert x into tree and return the index rather than having to search and insert separately
    #  what if we have same num in tree already...return highest index of that num..or have attribute for amount
    # insert item to tree
    def insert(self, item):
        node, found = self.search(item)
        # item already in tree, increment size, update #children attribute for O(logn) parents
        if found:
            node[6] += 1
            self.update_parents(self.tree[node[1]])
        else:  # have to add item as a child of given parent node
            self.tree[item] = [item, node[0], None, None, 0, 0, 1]
            if item < node[0]:
                # update parent's left child pointer and balance
                node[2] = item
                node[4] -= 1
            else:
                # update parent's right child pointer and balance
                node[3] = item
                node[4] += 1
            # check if we need to balance the tree
            self.update_parents(node)
        return

    # Given a parent node, B, perform rotations for parent nodes until tree is balanced, or handle updating all
    # parent node's children attributes until the root of the tree in the case of a
    def update_parents(self, node):
        # TODO: handle update children no rotate calls
        # move up O(logn) parents towards root, rotating any unbalanced nodes till we achieve balance
        left_child = self.tree.get(node[2])
        right_child = self.tree.get(node[3])
        while node:
            parent = self.tree.get(node[1])
            # left unbalanced
            if node[4] < -1:
                # TODO: update balance attributes
                # update parent to parent links to be parent to child
                left_child[1] = node[1]
                # change parent's attributes
                if child[1]:
                    # parent is a left child
                    if parent[2] == node[0]:
                        parent[2] = child[0]
                    elif parent[3] == node[0]:
                        parent[3] = child[0]
                else:
                    self.root = child[0]
                    # TODO: must update childs childs pointers and parents right child pointer
                    # and handle if there are none
                # change child's right child to be parent's left child
                node[2] = child[3]
                self.tree[child[3]][1] = node[0]
                # update child's right attributes
                child[3] = node[0]

            # right unbalanced
            elif node[4] > 1:
                self.left_rotate()
            # balanced, just updating children attribute
            else:

                pass
            # update # children up to root
            node[5] += 1
            # get next parent
            # TODO: maybe bug with passing node to function and then trying to change it?
            child = node
            node = parent
            # update child and parent balance attributes

            # child[4] = (self.tree[child[] + child[])

    #
    def left_rotate(self):
        return

    #
    def right_rotate(self):
        return

    #
    def remove(self, item):
        return


test()
