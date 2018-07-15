# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        current = [root]
        res = []
        nxt = []
        tmp_val = []
        while len(current) > 0:
            tmp = current.pop(0)
            tmp_val.append(tmp.val)
            if tmp.left is not None:
                nxt.append(tmp.left)
            if tmp.right is not None:
                nxt.append(tmp.right)
            if len(current) == 0:
                if nxt == []:
                    if tmp_val:
                        res.append(tmp_val)
                    return res
                else:
                    current = nxt
                    nxt = []
                res.append(tmp_val)
                tmp_val = []
        