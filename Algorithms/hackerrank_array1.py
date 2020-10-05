""" Hackerrank array problem: https://www.hackerrank.com/challenges/crush/problem
Starting with a 1-indexed array of zeros and a list of operations, for each operation add a
value to each of the array element between two given indices, inclusive.
Once all operations have been performed, return the maximum value in the array.

arrayManipulation has the following parameters:

    int n - the number of elements in the array
    int queries[q][3] - a two dimensional array of queries where each queries[i] contains three integers, a, b, and k.

m = number of queries, O(2*10^5)
n = number of array elements, O(10^7)
a,b = start,end indices O(n)
k = query addition value, O(10^9)
"""
from random import randint
import time


# faster method to solve problem, takes time: O(m + 2mlogm) = scan(m) + 2*sort(mlogm)
def find_max(n, queries):
    # TODO: do we have to consider cases of bad input such as start index < end index or negative values?
    # sort queries into two arrays with items (start/end index, value), for addition and subtraction, respectively
    # use a binary search to find insertion point or add items then sort in place?
    add_arr = [(x[0], x[2]) for x in queries]
    sub_arr = [(x[1], x[2]) for x in queries]
    add_arr.sort(key=lambda key: key[0])
    sub_arr.sort(key=lambda key: key[0])
    # now calculate best possible sum by iterating through ranges, adding value at start index and subtracting value at
    # end index using the two arrays we have constructed, keeping track of max sum as we go
    # TODO: simpler method to implement iteration? maybe use only 1 while loop and more conditionals
    curr_sum = 0
    max_sum = 0
    add_ind = 0
    sub_ind = 0
    m = len(add_arr)
    # only have to exhaust all possible additions to find max value, so max loop is size m rather than using while loop
    for _ in range(m):
        # inner loops process addition/subtraction, keep adding/subtracting from sum if the same array start/end index
        # is found at consecutive item in add/sub array, then
        for _ in range(m - add_ind):  # max loop is number of add indexes left
            curr_sum += add_arr[add_ind][1]
            if curr_sum > max_sum:
                max_sum = curr_sum
            # keep iterating till we reach a different start index, can then increment index and break loop
            add_ind += 1
            # we have finished checking all possible sums
            if add_ind == m:
                return max_sum
            # if next item is a new index, must check for subtractions first
            if add_arr[add_ind - 1][0] < add_arr[add_ind][0]:
                break
        # process subtraction now
        for _ in range(m - sub_ind):
            # if our current end index is before the next start index we can subtract, else move back to addition
            if sub_arr[sub_ind][0] < add_arr[add_ind][0]:
                curr_sum -= sub_arr[sub_ind][1]
                sub_ind += 1
            else:
                break


# generate random input for the problem
def random_input(m, n, k):
    queries = []
    for i in range(m):
        start = randint(0, n-1)
        end = randint(start, n-1)
        val = randint(0, k)
        queries.append((start, end, val))
    return queries


# solves the problem as stated, will take O(m*n) time, very slow
def naive(n, queries):
    array = [0 for _ in range(n)]
    max_val = 0
    for query in queries:
        for x in range(query[0] - 1, query[1]):
            array[x] += query[2]
            if array[x] > max_val:
                max_val = array[x]
    return max_val


def test():
    # q = random_input(20000, 10000000, 1000000000)
    file = open("hackerrank test data.txt", 'r')
    queries = []  # list of each puzzle in the file
    for line in file:
        # data is in the form: "start_index end_index val" eg. "0 10 985"
        sep1 = line.find(" ")
        sep2 = line.find(" ", sep1 + 1)
        start = int(line[:sep1])
        end = int(line[sep1 + 1:sep2])
        val = int(line[sep2 + 1:])
        # print(start, end, val)
        if start >= end:
            print("bad index")
        queries.append((start, end, val))
    file.close()
    timer = time.time_ns()
    print(find_max(0, queries))
    print("time taken: %f s" % ((time.time_ns() - timer) / 1000000000))
    # test case answer: 2497169732


test()
