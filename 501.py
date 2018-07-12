# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        d = {}

        def helper(root, d):
            if root is None:
                return
            helper(root.left, d)
            if root.val in d:
                d[root.val] += 1
            else:
                d[root.val] = 1
            helper(root.right, d)

        helper(root, d)
        l = list(d.items())
        l = sorted(l, key=lambda x: x[1], reverse=True)
        res = [l[0][0]]
        for i in range(1, len(l)):
            if l[i][1] == l[0][1]:
                res.append(l[i][0])
            else:
                break
        return res



