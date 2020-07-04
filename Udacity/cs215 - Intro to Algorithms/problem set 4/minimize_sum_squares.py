#
# Given a list of numbers, L, find a number, x, that
# minimizes the sum of the square of the difference
# between each element in L and x: SUM_{i=0}^{n-1} (L[i] - x)^2
#
# Your code should run in Theta(n) time
#


def minimize_square(L):
    # f(x) = sum(to n) (L[i] - x)**2
    # g(x) = x**2, h(x) = L[i] - x
    # g'(x) = 2x, h'(x) = -1
    # f'(x) = sum (g'(h(x)) . h'(x))
    #       = sum 2(L[i] - x).(-1)
    #       = sum -2(L[i] - x)
    # minimum f(x) is when f'(x) = 0, or minimise f(x) as f'(x) approaches 0
    # therefore: 0 = -2 sum(L[i] - x)
    #            0 = sum(L[i) - nx  (sum is same as adding all elements and subtracting x, n times)
    #           nx = sum L[i]
    #            x = (sum L[i])/n
    # This is the average value of the list, so to minimise f(x), find x that is closest to the average
    # can do this in two loops, 2n operations, Theta(n) time
    average = 0
    for item in L:
        average += item
    average /= len(L)
    print("average", average)
    smallest_diff = abs(average - L[0])
    x = 0
    for item in L:
        diff = abs(average - item)
        if diff < smallest_diff:
            x = item
            smallest_diff = diff
    return x
