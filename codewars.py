""" Codewars puzzles """


def langtons_ant(grid, column, row, n, direction=0):
    # 0,1,2,3 = north, east, south, west
    # Sets give the order of direction after rotating for counter clockwise and clockwise, respectively.
    # After the last direction is reached (a full rotation), the order resets to index 0
    rotation_order_sets = [(2, 1, 0, 3), (0, 1, 2, 3)]
    # Start moving the ant, run through n iterations
    for x in range(n):
        position_colour = grid[row][column]  # 1 is white, 0 is black
        # Get the rotation order corresponding to the current rotation direction, and determine the ants new direction
        rotation_order = rotation_order_sets[position_colour]
        direction_index = rotation_order.index(direction, 0) + 1
        # full rotation completed, must reset direction or index will be out of range
        if direction_index > 3:
            direction_index = 0
        # direction after rotation
        direction = rotation_order[direction_index]
        # change position to opposite colour
        grid[row][column] = 1 - position_colour

        # TODO: simple method to change direction...
        # if position_colour:  # white
        #     direction += 1
        #     if direction == 4:
        #         direction = 0
        # else:  # black
        #     direction -= 1
        #     if direction == -1:
        #         direction = 3
        
        # TODO: alternate method to determine direction using modulo
        # if position_colour == 1:  # white
        #     direction = (direction + 1) % 4
        # elif position_colour == 0:  # black
        #     direction = 4 - (direction - 1) % 4
        
        # TODO: 1 liners with more math, equivalent to the above operations with modulo
        # (1 - position_colour) leads to switching colour value: white (1) -> 0, black (0) -> 1
        # (2 * position_colour - 1) leads to increment/decrement: white(1) -> +1, black(0) -> -1
        # absolute value because 0 - (0-3) is negative
        # direction = abs(4 * (1 - position_colour) - ((direction + 2 * position_colour - 1) % 4))
        # +4 before % operation has no effect on the new direction after incr./decr. if it is 0-3 e.g. (4 + d=1) % 4 = 1
        # but if we increment to 4 or decrement to -1  we get the correct changes: (4 + d=4) % 4 = 0, (4 + d=-1) % 4 = 3 
        # direction = (4 + direction + 2 * position_colour - 1) % 4
        # direction = (3 + direction + 2 * position_colour) % 4
        
        # Change ants position
        if direction == 0:  # north
            row -= 1
        elif direction == 1:  # east
            column += 1
        elif direction == 2:  # south
            row += 1
        elif direction == 3:  # west
            column -= 1

        # Handle expanding the grid if the ant needs to move to a position outside the current grid
        # Also, if row/col = -1, change to 0 as the inserted group is now the first in the grid
        # TODO: why expand with black not white?
        num_columns = len(grid[0])
        num_rows = len(grid)
        if row == num_rows:  # add extra row at bottom of grid
            grid.append([0 for _ in range(num_columns)])
        elif row == -1:  # add extra row at top of grid
            grid.insert(0, [0 for _ in range(num_columns)])
            row = 0
        elif column == num_columns:  # add extra column to right side of grid
            for row_ in grid:  # add new item to the end of every row to create new column
                row_.append(0)
        elif column == -1:  # add extra column to left side of grid
            for row_ in grid:  # add new item to start of every row to create new column
                row_.insert(0, 0)
            column = 0
    return grid

# print(langtons_ant([[1,0], [1,1]], 0, 0, 5, 0))


from math import log, ceil
def population_growth(p0, percent, aug, p):
    # Math to solve for num years (n) given: p0, d = 1 + percent/100, aug (A), p (p(n))
    # population after n years:
    #              p(1) = p0*d + A
    #              p(2) = p1*d + A = (p0*d + A)*d + A = p0d^2 + Ad + A
    #              p(3) = p2*d + A = (p0d^2 + Ad + A)*d + A = p0d^3 + Ad^2 + Ad + A
    #              p(n) = p0d^n + SUM(k=0 -> n-1) Ad^(k)

    # For percent = 0, d = 1, therefore:
    #              p(n) = p0(1) + SUM(k=0 -> n-1) A(1)
    #              p(n) = p0 + nA
    #                 n = (p(n) - p0)/A

    # For d != 1, sum term is a geometric series: SUM(k=0 -> n-1) Ad^(k) = A(1 - d^n)/(1 - d)
    #              p(n) = p0d^n + A(1 - d^n)/(1 - d)
    #              p(n) = ((p0d^n - p0d*d^n) + (A - Ad^n))/(1 - d)
    #      p(n) - dp(n) = p0d^n - p0d*d^n + A - Ad^n
    #  p(n) - dp(n) - A = (p0 - p0d - A)d^n
    #               d^n = (p(n) - dp(n) - A)/(p0 - p0d - A)
    #          log(d^n) = log((p(n) - dp(n) - A)/(p0 - p0d - A))
    #           nlog(d) = log((p(n) - dp(n) - A)/(p0 - p0d - A))
    #                 n = log((p(n) - dp(n) - A)/(p0 - p0d - A)) / log(d)

    if percent == 0:
        years = (p - p0) / aug
    else:
        d = 1 + percent / 100
        if p == p0:
            return 0
        # terms inside logarithm can not be 0 or negative or gives error...
        # sub d = 1 + percent/100: (p - d*p - aug) = (p*percent + 100*aug), (p0 - p0*d - aug) = (p0*percent + 100*aug)
        # both log terms are always positive, must make exception for if p = p0  as term = 0 which gives error
        years = log((p - d * p - aug) / (p0 - p0 * d - aug)) / log(d)
        # years = (log(p*percent + 100*aug) - log(p0*percent + 100*aug)) / log(d)
    n = ceil(years)  # round up to nearest whole year
    return n


# print(population_growth(1500000, 0.0, 10000, 2000000))

def narcissistic(value):
    narc = 0
    string = str(value)
    for digit in string:
        narc += int(digit) ** len(string)
    return narc == value
    # return value == sum(int(x) ** len(str(value)) for x in str(value))


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

    # TODO: remove values on a1a2,b1b2,c1c2,d1d2 lines so we dont check them in next step?

    # TODO: simplify 1st sweep and region checks to 1 block each...

    for coords in coord_list:
        # a points
        if coords[1] > a[0][1]:
            a = [coords, coords]
        elif coords[1] == a[0][1]:
            # left most a value
            if coords[0] < a[0][0]:
                a[0] = coords
            # right most a value
            elif coords[0] > a[1][0]:
                a[1] = coords
        # c points
        if coords[1] < c[0][1]:
            c = [coords, coords]
        elif coords[1] == c[0][1]:
            # left most c value
            if coords[0] < c[0][0]:
                c[0] = coords
            # right most c value
            elif coords[0] > c[1][0]:
                c[1] = coords
        # b points
        if coords[0] > b[0][0]:
            b = [coords, coords]
        elif coords[0] == b[0][0]:
            # top most b value
            if coords[1] > b[0][1]:
                b[0] = coords
            # bottom most b value
            elif coords[1] < b[1][1]:
                b[1] = coords
        # d points
        if coords[0] < d[0][0]:
            d = [coords, coords]
        elif coords[0] == d[0][0]:
            # top most d value
            if coords[1] > d[0][1]:
                d[0] = coords
            # bottom most d value
            elif coords[1] < d[1][1]:
                d[1] = coords

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
            grads[index] = (pair[1][y2x2[index]][1] - pair[0][y1x1[index]][1]) / \
                           (pair[1][y2x2[index]][0] - pair[0][y1x1[index]][0])

    # TODO: How do I remove the full check for any region if flagged without checking the flag every loop...?
    # TODO: Without also having to write out code for every combination of flags...

    # determine the points in each region
    region1, region2, region3, region4 = [], [], [], []  # above a2b1, below b2c2, below c1d2, above d1a1
    for coords in coord_list:  # points on lines; ab, bc, cd, da, will fail at 2nd conditions
        if not flag_ab and a[1][0] < coords[0] < b[0][0] and grads[0] < ((coords[1] - a[1][1])/(coords[0] - a[1][0])):
            region1.append(coords)
        elif not flag_bc and c[1][0] < coords[0] < b[1][0] and grads[1] < ((coords[1] - b[1][1])/(coords[0] - b[1][0])):
            region2.append(coords)
        elif not flag_cd and c[0][0] > coords[0] > d[1][0] and grads[2] < ((coords[1] - c[0][1])/(coords[0] - c[0][0])):
            region3.append(coords)
        elif not flag_da and a[0][0] > coords[0] > d[0][0] and grads[3] < ((coords[1] - d[0][1])/(coords[0] - d[0][0])):
            region4.append(coords)

    # determine hulls for each region, first sort by x or y and then scan points checking x or y value and
    # gradients and accepting the point or rejecting and backtracking to last acceptable point

    # region 1: sort by descending y, gradient is decreasing
    hull = [a[1]]
    region1.sort(key=lambda vertex: vertex[1], reverse=True)
    prev_grad = 0
    counter = 0
    if a[1] != b[0]:
        region1.append(b[0])
    for point in region1:
        # reject point if its x value is less than the previous point, as it cannot be part of the hull
        # this means we are also limiting the tested gradients to -ve values due to the y sorting we have done
        if point[0] <= hull[-1][0]:  # also protects from zero division errors and duplicate points
            continue
        next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        # if we have same gradient as previous, remove the intermediate point as per the instructions
        if next_grad == prev_grad:
            hull.pop()
            hull.append(point)
            continue
        # must backtrack if our gradient increases, till we have consecutive decreasing gradients
        while next_grad > prev_grad:
            counter -= 1
            hull.pop()
            # if we backtrack to start of list, then we must stop trying to check previous gradients. and connect latest
            # point to starting point
            if counter == 0:
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])  # this will get set as prev grad
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        prev_grad = next_grad
        counter += 1

    # region 2: sort by descending x, gradient is decreasing
    region2.sort(key=lambda vertex: vertex[0], reverse=True)
    # make sure points are not the same before adding to hull
    if b[0] != b[1]:
        hull.append(b[1])
    if b[1] != c[1]:
        region2.append(c[1])
    # initialise loop this time as we cannot use negative infinity (idk how)
    if region2:
        prev_grad = (region2[0][1] - hull[-1][1]) / (region2[0][0] - hull[-1][0])
        counter = 1
        hull.append(region2[0])
    for point in region2:
        # reject higher or equal y values will also skip the first item which we already added
        if point[1] >= hull[-1][1]:
            continue
        # if zero division error, we have p1(y) > p2(y), p1(x) = p2(x), set inf gradient and let backtrack occur
        try:
            next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        except ZeroDivisionError:
            next_grad = math.inf
        # if we have same gradient as previous, remove the intermediate point as per the instructions
        if next_grad == prev_grad:
            hull.pop()
            hull.append(point)
            continue
        # must backtrack if our gradient increases, till we have consecutive decreasing gradients
        while next_grad > prev_grad:
            counter -= 1
            hull.pop()
            if counter == 0:
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        prev_grad = next_grad
        counter += 1

    # region 3: sort by ascending y, gradient is decreasing
    region3.sort(key=lambda vertex: vertex[1])
    # want to add in d2, but have to possibly delete later
    if c[0] != d[1]:
        region3.append(d[1])
    if c[0] != c[1]:
        hull.append(c[0])
    prev_grad = 0
    counter = 0
    for point in region3:
        # moving negative x direction, reject higher x values
        if point[0] >= hull[-1][0]:
            continue
        next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        # if we have same gradient as previous, remove the intermediate point as per the instructions
        if next_grad == prev_grad:
            hull.pop()
            hull.append(point)
            continue
        # must backtrack if our gradient increases, till we have consecutive decreasing gradients
        while next_grad > prev_grad:
            counter -= 1
            hull.pop()
            if counter == 0:
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        prev_grad = next_grad
        counter += 1

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
    prev_grad = math.inf
    counter = 0
    for point in region4:
        # lower y are ignored
        if point[1] <= hull[-1][1]:
            continue
        # if zero division error, we have p1(y) < p2(y), p1(x) = p2(x), set inf gradient and let backtrack occur
        try:
            next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        except ZeroDivisionError:
            next_grad = math.inf
        # if we have same gradient as previous, remove the intermediate point as per the instructions
        if next_grad == prev_grad:
            hull.pop()
            hull.append(point)
            continue
        # must backtrack if our gradient increases, till we have consecutive decreasing gradients
        while next_grad > prev_grad:
            counter -= 1
            hull.pop()
            if counter == 0:
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        prev_grad = next_grad
        counter += 1

    # must delete a1, if we placed it in region 4, and it overlaps with a2
    if hull[-1] == a[1]:
        hull.pop()

    print(hull)
    # returns a list of vertices on the convex hull starting from a2 and travelling clockwise around the polygon to a1
    return hull


# initial hull with only regions 2,4 and 3 vertically stacked points in each
convex_hull([[10,15], [10,10], [5,0], [0,0], [0,5], [5,15], [9,6], [9,5], [9,4], [2,11], [2,12], [2,13]])


