

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
