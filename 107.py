# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
	if root is None:
		return []
	p = [root]
	results = []
	current_level_num = 1
	next_level_num = 0
	d = []
	while p:
		current = p.pop(0)
		d.append(current.val)
		current_level_num -= 1
		if current.left != None:
			p.append(current.left)
			next_level_num += 1
		if current.right != None:
			p.append(current.right)
			next_level_num += 1
		if current_level_num == 0:
			current_level_num = next_level_num
			next_level_num = 0
			results.append(d)
			d = []
	return results[::-1]


