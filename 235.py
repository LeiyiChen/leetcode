# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        minn = min(p.val, q.val)
        maxn = max(p.val, q.val)
        if root is None:
        	return None
        if minn <= root.val <= maxn:
        	return root
        else:
        	l = lowestCommonAncestor(root.left, p, q)
        	r = lowestCommonAncestor(root.right, p, q)
        	if l:
        		return l
        	if r:
        		return r


