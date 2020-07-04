

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
