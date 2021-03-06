# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
        	return False
        sum-=root.val
        if sum == 0:
        	if root.left is None and root.right is None:
        		return True
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)