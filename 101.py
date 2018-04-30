# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def f(p, q):
        	if p == None:
        		return q == None
        	if q == None:
        		return p == None
        	if p.val == q.val:
        		return f(p.left, q.right) and f(p.right, q.left)
        	if p.val != q.val:
        		return False
        if root == None:
        	return True
        return f(root.left, root.right)

        



