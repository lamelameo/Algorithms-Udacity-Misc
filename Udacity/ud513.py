""" Code from UD513 """


""" Add three functions to the LinkedList.
"get_position" returns the element at a certain position.
The "insert" function will add an element to a particular
spot in the list.
"delete" will delete the first element with that
particular value.
Then, use "Test Run" and "Submit" to run the test cases
at the bottom."""


class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element

    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""
        current = self.head
        index = 1
        # iterate through till end of list
        while current:
            if index == position:
                return current
            else:
                current = current.next
                index += 1
        # made it through list without finding position
        return None

    def insert(self, new_element, position):
        """Insert a new node at the given position.
        Assume the first position is "1".
        Inserting at position 3 means between
        the 2nd and 3rd elements."""

        # handle special cases
        if position < 1:
            print("negative indexing not supported")
            return
        # adding to start of list
        elif position == 1:
            new_element.next = self.head
            self.head = new_element
            return
        # for anything else have to do some work - if prev node is None, we have been given an index which is out
        # of the list, if next node is None, we are adding to end of list
        prev_node = self.get_position(position - 1)
        if prev_node:
            next_node = prev_node.next
            if next_node:  # there is a next node, must add reference from new element to it
                new_element.next = next_node
            # in either case we must update prev nodes reference to be new element
            prev_node.next = new_element
        else:
            print("index out of list range")

    def delete(self, value):
        """Delete the first node with a given value."""
        current = self.head
        prev = None
        # iterate till we find value or reach end of list
        while current:
            # found value, must delete it and change references
            if current.value == value:
                if prev:  # item is not at start of list
                    prev.next = current.next
                else:  # item is first in list, must update head
                    self.head = current.next
                del current
                print("deleted value")
                return
            # continue iterating
            else:
                prev = current
                current = current.next
        print("value not found")


def binary_search(input_array, value):
    """You're going to write a binary search function.
    You should use an iterative approach - meaning
    using loops.
    Your function should take two inputs:
    a Python list to search through, and the value
    you're searching for.
    Assume the list only has distinct elements,
    meaning there are no repeated values, and
    elements are in a strictly increasing order.
    Return the index of value, or -1 if the value
    doesn't exist in the list."""
    lower = 0
    higher = len(input_array) - 1
    # loop till we find number or our bounds collapse, meaning number is not in array
    while lower <= higher:
        middle = int((higher + lower)/2)  # rounds down to get middle
        # check middle number
        item = input_array[middle]
        # found value
        if item == value:
            return middle
        # value less than item checked
        elif item > value:
            higher = middle - 1
        # value greater than item checked
        else:
            lower = middle + 1
    # made it through loop without finding value, it is not present in list
    return -1


def quicksort(array):
    """Implement quick sort in Python.
    Input a list.
    Output a sorted list."""

    # recurse to sort each range of indexes
    def recurse(low, high):
        # only have 1 item, no sorting needed, move on to other recursions
        if low == high:
            return
        # TODO: randomising pivot is better, but we will have to move to end of range for algorithm to work
        # pivot starts at upper bound and moves down if we do a swap
        pivot = high
        # item to check starts at lower bound and moves up if we dont do a swap
        check = low
        # compare all elements in range to pivot, swapping values where necessary
        for _ in range(high - low):
            # swap if pivot less than or equal to item checked, else leave pivot and increment check index
            if array[pivot] <= array[check]:
                temp = array[check]
                array[check] = array[pivot - 1]
                array[pivot - 1] = array[pivot]
                array[pivot] = temp
                pivot -= 1
            else:
                check += 1

        # once we have done all swaps for this range must recurse till low=high
        # if there are more items smaller than pivot, recurse with those items first, else use items which are greater
        # must also check if pivot is at lower or higher bounds, if so then we only check above or below, respectively
        if (pivot - low) > (high - pivot):  # more smaller items
            if pivot != high:
                recurse(pivot + 1, high)
            if pivot != low:
                recurse(low, pivot - 1)
        else:  # more greater items or same number of smaller and greater
            if pivot != low:
                recurse(low, pivot - 1)
            if pivot != high:
                recurse(pivot + 1, high)

    # start sort using bounds of array
    recurse(0, len(array)-1)
    return array


test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
test1 = [1, 1, 5, 4, 3, 2, 1, 1]
# print(quicksort(test1))


def bubblesort(array):
    # check through array n times, with largest element being placed at end each time, reducing search by 1 each time
    length = len(array)
    for x in range(length):
        print(x)
        swapped = False
        for index in range(length - 1 - x):
            if array[index] > array[index + 1]:
                # swap items without temporary variable, because why not
                array[index] = array[index] + array[index + 1]  # x1 = x0 + y0
                array[index + 1] = array[index] - array[index + 1]  # y1 = x1 - y0 = (x0 + y0) - y0 = x0
                array[index] = array[index] - array[index + 1]  # x1 = x1 - y1 = (x0 + y0) - x0 = y0
                print(array)
                swapped = True
        if not swapped:
            break
    return array


# bubblesort(test)
