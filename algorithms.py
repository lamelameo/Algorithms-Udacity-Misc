""" Algorithms I decided to code myself """

import timeit


# TODO: consider a heuristic algorithm using something like xy value in each quadrant, higher xy = more likely to be a
# TODO: point on the hull
def convex_hull(coord_list):
    import math
    # Sweep to determine max/min x,y points and the initial hull:  a1,a2 = y(max left,rightmost),
    # b1,b2 = x(max, up,downmost), c1,c2 = y(min left,rightmost), d1,d2 = x(min up,downmost)

    #       r4    a1---a2    r1
    #           x         x
    #         x             x
    #       d1               b1
    #        |               |
    #       d2               b2
    #         x             x
    #           x         x
    #       r3    c1---c2    r2

    # These points are on the convex hull, and give us regions to search for points in where any other points on
    # on the hull must be. These regions, r1,r2,r3,r4, are above a2b1 and d1a1, but below b2,c2 and c1d2.
    # We must do a sweep of the points and determine if any points are in these regions, which can be done
    # using the gradients of these lines and lines between the point and one on these lines (a1,b2,c1,d1)
    # If our region boundary lines are a single point, e.g. when a2 = b1, then we dont have to check it.
    # If we have any points in our regions, we must then determine if and how they are connected to the hull.
    # [[[ Here we could (but wont) use a practical adjustment such as max(x + y) in each region to get another point
    # (not necessarily on the hull) to sweep and eliminate more points before sorting if there are many points. ]]]
    # We will move clockwise around the hull/points, ie. from r1->r4. Sort points by x or y depending on the region and
    # check the gradient of consecutive points (starting at a2, b2, c1, d1). There will be a decreasing gradient across
    # every hull point. If the gradient increases, we remove the previous hull point then we must backtrack, checking
    # gradients of previous 2 points vs. prev point and current, until there is a decrease, or we hit the first point.
    # This step takes worst case 2p, if we have to add and backtrack all points, which is 2n for all points in regions.
    # Total time is 2n -> nlogn depending on how many hull vertices and how many points in our regions there are.
    # E.g. a sqaure takes 1 sweep (n) to find the initial hull, and determine there are no regions to check
    # If all points are part of convex hull, then time is: (sort) nlogn + (sweep) 2n + (construct) 2n

    # initialise [a1, a2], [b1, b2], [c1, c2], [d1, d2]
    a, b, c, d = [coord_list[0], coord_list[0]], [coord_list[0], coord_list[0]], \
                 [coord_list[0], coord_list[0]], [coord_list[0], coord_list[0]]

    # determine corner/border values; a,b,c,d
    def sort_coords(coords_, corners, bool1, bool2, bool3, bool4):
        # if current coord is further than current corner point in the direction we are checking then replace
        if bool1:
            corners[0], corners[1] = coords_, coords_
        # if current coord is same value in that direction, check the other direction to see if we have a new corner
        # ie an expanded border in this direction eg: a1-a2 -> a1----a2
        elif bool2:
            # check left/top direction (a1,b1,c1,d1)
            if bool3:
                corners[0] = coords_
            # check right/bottom direction (a2,b2,c2,d2)
            elif bool4:
                corners[1] = coords_

    for coords in coord_list:
        sort_coords(coords, a, coords[1] > a[0][1], coords[1] == a[0][1], coords[0] < a[0][0], coords[0] > a[1][0])
        sort_coords(coords, c, coords[1] < c[0][1], coords[1] == c[0][1], coords[0] < c[0][0], coords[0] > c[1][0])
        sort_coords(coords, b, coords[0] > b[0][0], coords[0] == b[0][0], coords[1] > b[0][1], coords[1] < b[1][1])
        sort_coords(coords, d, coords[0] < d[0][0], coords[0] == d[0][0], coords[1] > d[0][1], coords[1] < d[1][1])

    # Check for overlapping points: a2/b1, b2/c2, c1/d2, d1/a1. If we find a region is a single point, then flag it
    # to skip in next step. For squares or vertical/horizontal lines all regions will be flagged and thus skipped
    flag_ab, flag_bc, flag_cd, flag_da = False, False, False, False
    if a[1] == b[0]:
        flag_ab = True
    if b[1] == c[1]:
        flag_bc = True
    if c[0] == d[1]:
        flag_cd = True
    if d[0] == a[0]:
        flag_da = True

    # determine rise and runs and gradients
    grads = [None, None, None, None]
    # mapping order to index we need to let us use a single for loop to calc gradients
    y2x2 = [0, 1, 1, 0]
    y1x1 = [1, 1, 0, 0]
    for index, (pair, flag) in enumerate([((a, b), flag_ab), ((b, c), flag_bc), ((c, d), flag_cd), ((d, a), flag_da)]):
        if not flag:
            try:
                grads[index] = (pair[1][y2x2[index]][1] - pair[0][y1x1[index]][1]) / \
                               (pair[1][y2x2[index]][0] - pair[0][y1x1[index]][0])
            except ZeroDivisionError:
                grads[index] = math.inf

    # TODO: How do I remove the full check for any region if flagged without checking the flag every loop...?
    # TODO: Without also having to write out code for every combination of flags...

    # determine the points in each region - isnt possible to get zero division error as we restrict x values
    region1, region2, region3, region4 = [], [], [], []  # above a2b1, below b2c2, below c1d2, above d1a1
    for coords in coord_list:  # points on lines; ab, bc, cd, da, will fail at 2nd or 3rd conditions
        if not flag_ab and a[1][0] < coords[0] and b[0][1] < coords[1] \
                and grads[0] < ((coords[1] - a[1][1])/(coords[0] - a[1][0])):
            region1.append(coords)
        elif not flag_bc and c[1][0] < coords[0] and b[1][1] > coords[1] \
                and grads[1] < ((coords[1] - b[1][1])/(coords[0] - b[1][0])):
            region2.append(coords)
        elif not flag_cd and c[0][0] > coords[0] and d[1][1] > coords[1] \
                and grads[2] < ((coords[1] - c[0][1])/(coords[0] - c[0][0])):
            region3.append(coords)
        elif not flag_da and a[0][0] > coords[0] and d[0][1] < coords[1] \
                and grads[3] < ((coords[1] - d[0][1])/(coords[0] - d[0][0])):
            region4.append(coords)

    # determine hulls for each region, first sort by x or y and then scan points checking x or y value and
    # gradients and accepting the point or rejecting and backtracking to last acceptable point
    def check_region(region, region_num, prev_grad_, counter_):
        for point in region:
            # skip points based on x or y values since we have sorted each region
            # and can
            index_ = region_num % 2
            if region_num == 0 or region_num == 3:
                if point[index_] <= hull[-1][index_]:
                    continue
            else:
                if point[index_] >= hull[-1][index_]:
                    continue

            # get gradient from previous to current point
            try:
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
            # we have vertically stacked colinear points, set infinite gradient to let backtrack occur
            except ZeroDivisionError:
                next_grad = math.inf

            # if we have same gradient as previous, remove the intermediate point as per the instructions
            if next_grad == prev_grad_:
                hull.pop()
                hull.append(point)
                continue
            # must backtrack if our gradient increases, till we have consecutive decreasing gradients
            while next_grad > prev_grad_:
                counter_ -= 1
                hull.pop()
                # if we backtrack to start of list, then we must stop trying to check previous gradients.
                # and connect latest point to starting point
                if counter_ == 0:
                    next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])  # this will get set as prev grad
                    break
                else:
                    prev_grad_ = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                    next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
            hull.append(point)
            prev_grad_ = next_grad
            counter_ += 1

    # region 1: sort by descending y, gradient is decreasing
    region1.sort(key=lambda vertex: vertex[1], reverse=True)
    hull = [a[1]]
    if a[1] != b[0]:
        region1.append(b[0])
    check_region(region1, 0, 0, 0)

    # region 2: sort by descending x, gradient is decreasing
    region2.sort(key=lambda vertex: vertex[0], reverse=True)
    # make sure points are not the same before adding to hull
    if b[0] != b[1]:
        hull.append(b[1])
    if b[1] != c[1]:
        region2.append(c[1])
    # initialise loop this time as we cannot use negative infinity (idk how)
    prev_grad, counter = None, 0
    if region2:
        try:
            prev_grad = (region2[0][1] - hull[-1][1]) / (region2[0][0] - hull[-1][0])
        except ZeroDivisionError:
            prev_grad = math.inf
        counter = 1
        hull.append(region2[0])
    check_region(region2, 1, prev_grad, counter)

    # region 3: sort by ascending y, gradient is decreasing
    region3.sort(key=lambda vertex: vertex[1])
    # want to add in d2, but have to possibly delete later
    if c[0] != d[1]:
        region3.append(d[1])
    if c[0] != c[1]:
        hull.append(c[0])
    check_region(region3, 2, 0, 0)

    # region 4: sort by ascending x, gradient is decreasing
    region4.sort(key=lambda vertex: vertex[0])
    # must delete d2 now, as overlaps with a2 which was already placed at start
    if d[1] == a[1]:
        hull.pop()
    # check there is a region then can add a1 and potentially delete later
    elif a[0] != d[0]:
        region4.append(a[0])
    if d[1] != d[0] and d[0] != a[1]:
        hull.append(d[0])
    check_region(region4, 3, math.inf, 0)

    # must delete a1, if we placed it in region 4, and it overlaps with a2
    if hull[-1] == a[1]:
        hull.pop()

    # returns a list of vertices on the convex hull starting from a2 and travelling clockwise around the polygon to a1
    return hull


# find the index where an item would be placed in a sorted array
def binary_search(array, item):
    first = 0
    last = len(array) - 1
    flag = None
    # loop till first = last, then compare if our value should be placed before or after it
    while first <= last:
        mid = (last + first)//2  # take ceil of floats
        if item < array[mid]:  # less than mid value
            last = mid - 1
            flag = 0
        elif item > array[mid]:  # greater than mid value
            first = mid + 1
            flag = 1
    if flag:  # insert after value
        return last + 1
    else:  # insert before value
        return last


# Find longest palindrome given a sequence of letters. We will move left to right across letters and searching for
# that letter's duplicate, if any, searching right to left and taking the first match we find. We will then recurse on
# the subsequence contained between these letters and also iterate over all starting letters left to right at
# all recursion depths. We will take the max of this iteration as our return result at each depth. If we recurse on a
# 1 or 2 length sequence, then this will be our starting result to return and move up one depth level.
def palindrome(word):
    # TODO: can use a data structure to hold each letter's duplicate's indexes so we dont have to search or less search
    # may need a lot of space to store and time to create?
    # TODO: can save palindromes themselves in dict under own key, rather than just the subsequences
    # this may save time if we find that palindrome, but requires extra space
    memo = dict()
    counter = 0
    duplicates = dict()
    # TODO: making array to make searching easier
    for index, character in enumerate(word):
        if character not in duplicates:
            duplicates[character] = [index]
        else:
            duplicates[character].append(index)

    # recursive function to find longest palindrome in a given sequence
    def recurse(seq):
        nonlocal counter
        # already stored value of this subsequence
        counter += 1
        if seq in memo:
            return memo[seq]

        # otherwise must find nested duplicate letters recursively to build up palindromes
        if len(seq) > 2:
            # if our outer letters are same, we can just recurse on inner letter and add outer to return value
            if seq[0] == seq[-1]:
                pal = recurse(seq[1:-1])
                memo[seq] = seq[0] + pal + seq[0]
                return memo[seq]
            # otherwise we must search for duplicate letters inside the subsequence to find best inner palindrome
            else:
                longest = ""
                # iterate through all letters as starting points to find longest palindrome
                for index_start, letter in enumerate(seq):
                    # find duplicate by iterating backward through sequence until the letter itself
                    for index_end in range(1, len(seq) - index_start):
                        # found duplicate and thus a palindrome
                        if seq[0 - index_end] == letter:
                            # recurse on subsequence
                            if index_end == 1:
                                pal = recurse(seq[index_start:])
                            else:
                                pal = recurse(seq[index_start:1 - index_end])
                            # TODO: * had palindrome saver here
                            break
                    else:  # no duplicates, recurse will just return the letter and memoize if necessary
                        pal = recurse(letter)

                    # TODO: This saves another entry for the palindrome itself, rather than sequence
                    # TODO: may save time but requires more space, is it useful?
                    # if pal not in memo:  # store if we havent seen this yet
                    #     print('test1:', pal)
                    #     memo[pal] = pal
                    # else:
                    #     print('test2:', pal)
                    #     pass

                    # obtain max length palindrome from all letters as starting positions
                    if len(pal) > len(longest):
                        longest = pal

                memo[seq] = longest
                return longest

        # must check for length 1 or 2 words, in case we start with these
        elif len(seq) == 2:  # word 2 letters
            if seq[0] == seq[1]:  # same letter
                memo[seq] = seq
                return seq
            else:  # different letters, must return only one letter so just use the first
                memo[seq] = seq[0]
                return seq[0]

        else:  # word = 1 letter (or "" ie no sequence)
            memo[seq] = seq
            return seq

    longest_palindrome = recurse(word)
    # print("memoized:", len(memo))
    # print("recursions:", counter)
    return longest_palindrome


# print("Longest palindrome:", palindrome("aaaaaaaaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaaa"), "\n")


# more simple method to get longest palindrome subsequence
def palindrome2(word):
    # store results as we determine them
    store = dict()
    counter = 0

    def recurse(seq):
        nonlocal counter
        counter += 1
        # print("sub sequence:", seq)
        # memoized already
        if seq in store:
            return store[seq]

        # length 1 subsequence
        if len(seq) == 1:
            store[seq] = seq
            return seq

        # length 2 subsequence
        elif len(seq) == 2:
            if seq[0] == seq[1]:
                store[seq] = seq
                return seq
            else:
                store[seq] = seq[0]
                return seq[0]

        # length > 2 subsequence
        else:
            # outer letters are a palindrome, must recurse and then add the result
            if seq[0] == seq[-1]:
                inner = recurse(seq[1:-1])
                store[seq] = seq[0] + inner + seq[0]
                return store[seq]
            # outer letters are different, recurse by incrementing inwards one letter on either side and take best
            else:
                left = recurse(seq[:-1])
                right = recurse(seq[1:])
                best = max(left, right, key=lambda k: len(k))
                store[seq] = best
                return best

    # recurse on word to get palindrome
    pal = recurse(word)
    # print('memoized:', len(store))
    # print('recursions:', counter)
    return pal


def test_palindrome():

    # print("Longset palindrome:", palindrome2('aaaaaaaaaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaa'))
    # abcdefghijklmnopqrstuvwxyzyxwvutrqponmlkjihgfedcba aaaaaaaaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaaa
    # abcabccbacbcabcabcabcabacbcbacbcabcabcbaabca aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaa

    words = "tt"
    extra = "a"
    setup1 = "from __main__ import palindrome"
    code1 = "palindrome(\"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\")"
    time = timeit.timeit(setup=setup1, stmt=code1, number=1000)
    print(time)

    # for x in range(22):
    #     words = extra + words + extra
    #     setup1 = "from __main__ import palindrome2"
    #     code1 = "palindrome2(\"{0}\")".format(words)
    #     print(code1)
    #     time = timeit.timeit(setup=setup1, stmt=code1, number=1)
    #     print(time)


# return dot product of two same sized matrices
def dot_product(m1, m2):
    result = []
    # iterate through all rows
    size_row = len(m1[0])
    for row_index, row in enumerate(m1):
        # print("row:", row)
        result.append([])
        # iterate through all items in row, multiplying by corresponding column value and adding to resulting matrix
        # item sum
        for m1_index, row_item in enumerate(row):
            # print('m1 item:', row_item)
            # iterate through all items in corresponding column
            for item_index in range(size_row):
                # print("m2 item:", m2[row_index][item_index])
                try:
                    result[row_index][item_index] += row_item*m2[m1_index][item_index]
                except IndexError:
                    result[row_index].append(row_item*m2[m1_index][item_index])
            # print('result:', result[row_index])
    return result


def test_dot_product():
    from math import inf
    test = [[1,2,3],[2,0,2],[1,2,1]]
    matrix = [[0,1,3,inf,6,inf,inf],
              [1,0,1,inf,inf,inf,inf],
              [3,1,0,2,inf,inf,inf],
              [inf,inf,2,0,1,2,inf],
              [6,inf,inf,1,0,inf,3],
              [inf,inf,inf,2,inf,0,1],
              [inf,inf,inf,inf,3,1,0]]

    print(dot_product(matrix, matrix))


