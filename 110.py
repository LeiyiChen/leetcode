# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def Depth(root):
        	if root is None:
        		return 0
        	elif root.left i None and root.right is None:
        		return 1
        	else:
        		return 1 + max(Depth(root.left), Depth(root.right))

        if root is None:
        	return True
        if abs(Depth(root.left) - Depth(root,right)) > 1:
        	return False
        else:
        	return isBalanced(root.left) and isBalanced(root.right)