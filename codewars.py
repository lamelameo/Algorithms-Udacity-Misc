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

# An algorithm to solve the convex hull problem in nlogn + (n-3) + 2(n-3) time in the worst case (all points are in
# regions above our starting hull, therefore sort all points by y value in nlogn + extra work to make hull
# Best case: 2n, if our starting hull is the convex hull, and rest of the points lie inside
# Firstly, do a single sweep of all points and determine the maximum and minimum x and y value points and mark each.
# Determine the initial hull with points: a = y(max), b = x(max), c = y(min), d = x(min), calc gradients: ab, bc, cd, da
# Next, do a single sweep to sort all non marked point into 3-4 regions (if we have only 3 points from first sweep ie
# two of the max/mins overlap, then 3 points for our starting hull). The regions are: (top right) above ab,
# (bottom right) below bc, (bottom left) below cd, (top left) above da. Any on these lines must be considered.
# Must consider special cases: any initial hull lines are parallel to the x or y axis, multiple initial hull points are
# max/mins (example a rectangular hull), 3 initial hull points instead of 4 (and therefore 3 regions not 4), all points
# lie on a single line

def convex_hull(coord_list):
    import math
    # takes a list of coordinate lists, [[x,y],...] and outputs a sublist of coordinate lists containing points that
    # make up the convex hull

    # Sweep to determine max/min x,y points and the initial hull:  a = y(max), b = x(max), c = y(min), d = x(min)
    # Also track left/right a,c points and top/bottom most b,d points
    a, b, c, d = [[coord_list[0]], [coord_list[0]]], [[coord_list[0]], [coord_list[0]]], \
                 [[coord_list[0]], [coord_list[0]]], [[coord_list[0]], [coord_list[0]]]

    # TODO: remove values on a1a2,b1b2,c1c2,d1d2 lines so we dont check them in next step?
    for coords in coord_list:
        if coords[1] > a[0][1]:
            a = [[coords], [coords]]
        elif coords[1] == a[0][1]:
            # right most a value
            if coords[0] > a[0][0]:
                a[0] = coords
            # left most a value
            elif coords[0] < a[1][0]:
                a[1] = coords

        if coords[1] < c[0][1]:
            c = [[coords], [coords]]
        elif coords[1] == c[0][1]:
            # right most c value
            if coords[0] > c[0][0]:
                c[0] = coords
            # left most c value
            elif coords[0] < c[1][0]:
                c[1] = coords

        if coords[0] > b[0][0]:
            b = [[coords], [coords]]
        elif coords[0] == b[0][0]:
            # top most b value
            if coords[1] > b[0][1]:
                b[0] = coords
            # bottom most b value
            elif coords[1] < b[1][1]:
                b[1] = coords

        if coords[0] < d[0][0]:
            d = [[coords], [coords]]
        elif coords[1] == d[0][0]:
            # top most d value
            if coords[1] > d[0][1]:
                d[0] = coords
            # bottom most d value
            elif coords[1] < d[1][1]:
                d[1] = coords

    # check for overlapping points in a,b,c,d, if we find a region is a single point, then flag it to skip in next step
    # for squares or vertical/vertical lines all regions will be flagged and thus skipped
    flag_ab, flag_bc, flag_cd, flag_da = False, False, False, False
    # unflagged = [1, 2, 3, 4]
    if a[1] == b[0]:
        flag_ab = True
        # unflagged.remove(1)
    if b[1] == c[1]:
        flag_bc = True
        # unflagged.remove(2)
    if c[0] == d[1]:
        flag_cd = True
        # unflagged.remove(3)
    if d[0] == a[0]:
        flag_da = True
        # unflagged.remove(4)

    # TODO: can check if we have a square or line here by checking for all empty regions then checking some stuff...

    # determine rise and runs and gradients, accounting for infinite gradients
    ab_rise = (b[0][1] - a[1][1])
    ab_run = (b[0][0] - a[1][0])
    bc_rise = (c[1][1] - b[1][1])
    bc_run = (c[1][0] - b[1][0])
    cd_rise = (d[1][1] - c[0][1])
    cd_run = (d[1][0] - c[0][0])
    da_rise = (a[0][1] - d[0][1])
    da_run = (a[0][0] - d[0][0])
    ab, bc, cd, da = 0, 0, 0, 0
    rises = [ab_rise, bc_rise, cd_rise, da_rise]
    runs = [ab_run, bc_run, cd_run, da_run]
    gradients = [ab, bc, cd, da]
    for x in range(4):
        try:
            gradients[x] = rises[x]/runs[x]
        except ZeroDivisionError:
            gradients[x] = math.inf

    # TODO: How do I remove the full check for any region if flagged without checking the flag every loop...?
    # TODO: Without also having to write out code for every combination of flags...

    # determine the points in each region
    region1, region2, region3, region4 = [], [], [], []  # above a2b1, below b2c2, below c1d2, above d1a1
    for coords in coord_list:  # points on lines; ab, bc, cd, da, will fail at 2nd conditions
        if not flag_ab and a[1][0] < coords[0] < b[0][0] and ab < ((coords[1] - a[1][1])/(coords[0] - a[1][0])):
            region1.append(coords)
        elif not flag_bc and c[1][0] < coords[0] < b[1][0] and bc < ((coords[1] - b[1][1])/(coords[0] - b[1][0])):
            region2.append(coords)
        elif not flag_cd and c[0][0] > coords[0] > d[1][0] and cd < ((coords[1] - c[0][1])/(coords[0] - c[0][0])):
            region3.append(coords)
        elif not flag_da and a[0][0] > coords[0] > d[0][0] and da < ((coords[1] - d[0][1]) / (coords[0] - d[0][0])):
            region4.append(coords)

    # determine hulls for each region, first sort by x or y and then scan points checking gradients

    # region 1: sort by descending y, gradient is decreasing
    hull = [a[1]]
    region1.sort(key=lambda vertex: vertex[1], reverse=True)
    prev_grad = 0
    counter = 0
    for point in region1:
        # reject point if its x value is less than the previous point, as it cannot be part of the hull
        # this means we are also limiting the tested gradients to -ve values due to the y sorting we have done
        if point[0] <= hull[-1][0]:  # protects from infinite gradients
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
                prev_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        counter += 1

    # region 2: sort by descending x, gradient is decreasing
    region2.sort(key=lambda vertex: vertex[0], reverse=True)
    hull.append(b[0])
    hull.append(b[1])
    # initialise loop this time as we cannot use negative infinity (idk how)
    prev_grad = (region2[0][1] - hull[-1][1]) / (region2[0][0] - hull[-1][0])
    counter = 1
    hull.append(region2[0])
    for point in region2:
        # reject higher y values
        if point[1] >= hull[-1][1]:  # will skip the first item
            continue
        next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        # if we have same gradient as previous, remove the intermediate point as per the instructions
        if next_grad == prev_grad:
            hull.pop()
            hull.append(point)
            continue
        # must backtrack if our gradient decreases, till we have consecutive increasing gradients
        while next_grad > prev_grad:
            counter -= 1
            hull.pop()
            if counter == 0:
                prev_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        counter += 1

    # region 3: sort by ascending y, gradient is decreasing
    region3.sort(key=lambda vertex: vertex[1])
    hull.append(c[1])
    hull.append(c[0])
    prev_grad = 0
    counter = 0
    for point in region3:
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
                prev_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        counter += 1

    # region 4: sort by ascending x, gradient is decreasing
    region4.sort(key=lambda vertex: vertex[0])
    hull.append(d[1])
    hull.append(d[0])
    prev_grad = 0
    counter = 0
    for point in region4:
        # lower y are ignored
        if point[1] <= hull[-1][1]:
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
                prev_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
                break
            else:
                prev_grad = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
                next_grad = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
        hull.append(point)
        counter += 1
    # append a1 as the last hull vertex
    hull.append(a[0])

    # returns a list of vertices on the convex hull starting from a2 and travelling clockwise around the polygon to a1
    return hull
