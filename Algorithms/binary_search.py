

# find the index where an item would be placed in a sorted array
def binary_search(array, item):
    first = 0
    last = len(array) - 1
    # loop till first < last, then take first as our index, as in all cases this is right choice
    # also should handle duplicates, returning index after last duplicate
    while first <= last:
        mid = (last + first)//2  # take floor of floats
        if item < array[mid]:  # less than mid value
            last = mid - 1
        else:  # greater than or equal to mid value
            first = mid + 1
    return first
