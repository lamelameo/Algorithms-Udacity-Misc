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

