# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def depth(p):
        	if p == None:
        		return 1
        	else:
        		return 1 + max(depth(p.right), depth(q.left))
        
        if root == None:
        	return 0
        else:
        	return max(depth(root.right), depth(root.left))
