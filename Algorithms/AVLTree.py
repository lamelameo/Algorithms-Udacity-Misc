""" Implementation of an AVL Tree using python dicts for pointer like connections.

    Supports duplicate values and the following operations:
    - Insert in O(logn)
    - Delete in O(logn)
    - Rank in O(logn)
    - Search in O(logn)
    - Successor in O(logn)

"""


class AVLTree:
    """ AVL tree data structure supporting O(logn) search, insert, and delete queries in O(n) space
    Takes one optional parameter: array - a List object which will be converted to a tree
    Each node in the tree is a list containing 7 items: (key, parent, left, right, height, children, size)
    Using the children attribute we can determine the rank of item in sorted list
    The size attribute stores the number of this key that are present in the set
     """
    def __init__(self, array=None):
        # TODO: use dict for pointer like functionality? This also means we have O(1) search not O(logn)??
        #  are there losses for dict for add/del - preprocess dict with array item with empty tuples?
        #  Can change dict syntax to something nicer?
        # self.tree = []
        self.tree = dict()
        self.root = None
        self.size = 0
        # we have been given an array to convert to a tree
        if array:
            self.root = array[0]
            self.tree[array[0]] = [array[0], None, None, None, 0, 0, 1]
            for item in array[1:]:
                self.insert(item)

    # return a string representation of the tree, successive line representing successive heights
    def __str__(self):
        tree = ""
        queue = [self.root]
        while queue:
            tree += str(queue.pop())
            while True:
                pass

        return tree

    # find the parent node to which the item should be attached, or the item's node if it already exists
    # return a tuple, tuple[0] is the node key and tuple[1] is 1 or 0 representing item node or parent node
    def search(self, item):
        node = self.tree.get(item)
        if node:
            return node, 1
        else:
            current = self.tree.get(self.root)
            # tree is empty
            if not current:
                return None, 0
            # traverse tree
            while True:
                if item < current[0]:  # item less than node - go left
                    if current[2]:
                        current = self.tree[current[2]]
                    else:
                        return current, 0
                else:  # go right
                    if current[3]:
                        current = self.tree[current[3]]
                    else:
                        return current, 0

    # TODO: insert x into tree and return the index rather than having to search and insert separately
    #  what if we have same num in tree already...return highest index of that num..or have attribute for amount
    # insert item to tree
    def insert(self, item):
        self.size += 1
        node, found = self.search(item)
        # inserting into empty tree
        if not node:
            self.root = item
            self.tree[item] = [item, node[0], None, None, 0, 0, 1]
            return
        # item already in tree, increment size, update #children attribute for O(logn) parents
        if found:
            node[6] += 1
            self.rebalance(self.tree.get(node[1]))
        else:  # have to add item as a child of given parent node
            self.tree[item] = [item, node[0], None, None, 0, 0, 1]
            if item < node[0]:
                # update parent's left child pointer
                node[2] = item
            else:
                # update parent's right child pointer
                node[3] = item
            # check if we need to balance the tree
            self.rebalance(node)

    # convenient function to call to deal with height of a node taking into account node=None
    def height(self, node):
        if not node:
            return -1
        else:
            return node[4]

    # update height and children attributes of a given node
    def update_height(self, node):
        left = self.tree.get(node[2])
        right = self.tree.get(node[3])
        node[5] = 0
        if not right:
            right = 0
        else:
            node[5] += right[6] + right[5]
        if not left:
            left = 0
        else:
            node[5] += left[6] + left[5]
        node[4] = 1 + max(self.height(left), self.height(right))

    # TODO: need to deal with changes to root node in rebalance, l/r rotate, delete

    # (key, parent, left, right, height, children, size)
    def rebalance(self, node):
        while node:
            self.update_height(node)
            #
            left = self.tree.get(node[2])
            right = self.tree.get(node[3])
            # left unbalanced
            if self.height(left) >= 2 + self.height(right):
                if self.height(self.tree.get(left[2])) >= self.height(self.tree.get(left[3])):
                    self.right_rotate(node)
                else:
                    self.left_rotate(left)
                    self.right_rotate(node)
            # right unbalanced
            elif self.height(right) >= 2 + self.height(left):
                if self.height(self.tree.get(right[3])) >= self.height(self.tree.get(right[2])):
                    self.left_rotate(node)
                else:
                    self.right_rotate(right)
                    self.left_rotate(node)
            node = self.tree.get(node[1])

    #
    def left_rotate(self, a):
        b = self.tree.get(a[3])
        alpha = self.tree.get(a[2])
        beta = self.tree.get(b[2])
        gamma = self.tree.get(b[3])
        parent = self.tree.get(a[1])
        # if A has a parent, change its left/right child from A to B and update parent of B
        if parent:
            if parent[0] > a[0]:
                parent[2] = b[0]
            else:
                parent[3] = b[0]
            b[1] = parent[0]
        else:
            b[1] = None
            self.root = b[0]
        # switch beta to be child of A and then switch A to be left child of B
        if beta:
            a[3] = beta[0]
            beta[1] = a[0]
        else:
            a[3] = None

        a[1] = b[0]
        b[2] = a[0]
        # update heights of A then B
        self.update_height(a)
        self.update_height(b)

    #
    def right_rotate(self, b):
        a = self.tree.get(b[2])
        alpha = self.tree.get(a[2])
        beta = self.tree.get(a[3])
        gamma = self.tree.get(b[3])
        parent = self.tree.get(b[1])
        # if B has a parent, change its left/right child from and update parent of A
        if parent:
            if parent[0] > b[0]:
                parent[2] = a[0]
            else:
                parent[3] = a[0]
            a[1] = parent[0]
        else:
            a[1] = None
            self.root = a[0]
        # switch beta to be child of B and then switch B to be right child of A
        if beta:
            b[2] = beta[0]
            beta[1] = b[0]
        else:
            b[3] = None

        b[1] = a[0]
        a[3] = b[0]
        # update heights of B then A
        self.update_height(b)
        self.update_height(a)

    # remove a node from the tree and rebalance
    # 4 possible cases:
    #   - Node has duplicate values, so decrement size attribute, no deletion
    #   (No Duplicate values)
    #   - 2 children; find the successor of the node and use this as its replacement before deleting
    #   - 1 child; use this child as the node's replacement before deleting
    #   - 0 children; delete the node
    def remove(self, item):
        # TODO: simplify code, probably can amalgamate lots of the branches
        node = self.tree.get(item)
        if not node:
            print("Item not found.")
            return
        self.size -= 1

        # handle updating links between parent and replacement for deleted node
        def modify_parent(self_, child):
            # update deleted nodes parent link
            if parent:
                # node is a left child
                if parent[2] == node[0]:
                    parent[2] = child[0]
                else:
                    parent[3] = child[0]
                # update left parent link
                child[1] = parent[0]
            else:  # deleted node was the root
                self_.root = child[0]
                child[1] = None

        # if duplicates only have to change item size attribute and update parent children attribute
        if node[6] > 1:
            node[6] -= 1
        else:
            # handle node with 0,1,2 children
            parent = self.tree.get(node[1])
            # node has left child
            if node[2]:
                left = self.tree[node[2]]
                # 2 children - get successor and replace
                if node[3]:
                    succ = self.successor(node)
                    print("succ", succ)
                    succ_par = self.tree.get(succ[1])
                    succ_child = self.tree.get(succ[3])
                    # successor has a right child
                    if succ_child:
                        # replace successor with its child - 2 cases
                        # successor is deleted nodes right child
                        if succ_par == node:
                            succ_par[3] = succ_child[0]
                            succ_child[1] = succ_par[0]
                        # successor is a left child of some node in the right sub tree of deleted node
                        else:
                            succ_par[2] = succ_child[0]
                            succ_child[1] = succ_par[0]
                        # replace node to be deleted with successor
                        succ[3] = node[3]
                        self.tree[node[3]][1] = succ[0]
                        succ[2] = left[0]
                        left[1] = succ[0]
                        # update deleted nodes parent link
                        modify_parent(self, succ)
                    # successor has no child
                    else:
                        # update deleted nodes parent link
                        modify_parent(self, succ)
                        # update left child link
                        succ[2] = left[0]
                        left[1] = succ[0]
                        # if successor was right child of deleted node, we are done, otherwise must update right child
                        # and parent of the successor
                        if succ_par != node:
                            succ[3] = node[3]
                            self.tree[node[3]][1] = succ[0]
                            succ_par[2] = None
                    node = succ
                # one (left) child
                else:
                    # update deleted nodes parent link
                    modify_parent(self, left)
                    node = left
            else:
                # only one (right) child
                if node[3]:
                    right = self.tree[node[3]]
                    # update deleted nodes parent link
                    modify_parent(self, right)
                    node = right
                # 0 children
                else:
                    # update deleted nodes parent link
                    modify_parent(self, [None, None])
                    node = None
            # delete item from dict
            self.tree.__delitem__(item)
        # rebalance the tree
        self.rebalance(node)

    # return the index the given item would be placed in a sorted array of the items in the tree
    def rank(self, item):
        # TODO: find items place in tree, then climb up tree to root counting number of nodes smaller than item by
        #  looking at parent size + left child at each step up we moved to a parent with a smaller value
        count = 0
        node, flag = self.search(item)
        parent = None
        # found item or item larger than insert parent node, count then start loop
        if flag or (node[0] < item):
            count += node[6]
            left = self.tree.get(node[2])
            if left:
                count += left[5] + left[6]
            parent = self.tree.get(node[1])

        while parent:
            # only count if we move up to a parent that is less than node
            if parent[0] < node[0]:
                count += parent[6]
                left = self.tree.get(parent[2])
                if left:
                    count += left[5] + left[6]
            # update node and parent
            node = parent
            parent = self.tree.get(parent[1])

        return count

    # TODO: what if we return None, ie no successor?
    # get the next highest item in sorted list
    def successor(self, node):
        # if we have a right subtree, find minimum in that tree
        right = self.tree.get(node[3])
        current = node
        if right:
            # succ = self.search(node[0] + 1)  # shortcut
            left = self.tree.get(right[2])
            while left:
                current = left
                left = self.tree.get(current[2])
            return current
        else:
            # if no right subtree, must move up to parents until root or until a parent is greater than current node
            # to get the next highest node
            # TODO: check this
            parent = self.tree.get(node[1])
            if not parent:
                return None
            while parent[3] == current[0]:
                current = parent
                parent = self.tree.get(current[1])
                # node was largest in the tree
                if not parent:
                    break
            return parent


def test():
    # TODO: need to check different conditions for rotations, successor, rank, insert, remove
    array = [1, 2, 3, 4, 3, 3, 5, 6, 7, 1, 2, 4]
    tree = AVLTree(array)
    print(tree.tree)
    print(tree.rank(4))
    tree.remove(4)
    tree.remove(3)
    print(tree.tree)
    tree.remove(4)
    print(tree.root)
    print(tree.tree)


test()
