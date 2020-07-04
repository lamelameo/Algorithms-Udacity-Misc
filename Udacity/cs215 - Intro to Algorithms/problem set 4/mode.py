#
# Given a list L of n numbers, find the mode
# (the number that appears the most times).
# Your algorithm should run in Theta(n).
# If there are ties - just pick one value to return
#


def mode(L):
    # use a dictionary to save the frequencies of each item using the item as the key
    # increasing frequency by 1 if the key already exists, or making a value of 1 if not
    # then check each key in the dictionary (which could be same length as list or a lot smaller)
    # keep track of the key for highest value seen so far, once checked all frequencies, then the key
    # we have is the item (or one of multiple) in the list with most frequency ie the mode

    frequencies = {}
    _mode = 0
    for item in L:
        if item not in frequencies:
            frequencies[item] = 1
        else:
            frequencies[item] += 1
            if frequencies[item] > _mode:
                _mode = frequencies[item]
    return _mode

    # TODO: checks all frequencies, to find the highest, slower but could list all highest or top x highest etc..
    # initialise the mode key arbitrarily to first item in list (it will be updated if it isnt the mode)
    # mode_key = L[0]
    # for value in frequencies:
    #     # determine if the current highest is lower than the value we are currently checking
    #     if frequencies[value] > frequencies[mode_key]:
    #         mode_key = value
    # return mode_key
