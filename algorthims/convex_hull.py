

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
