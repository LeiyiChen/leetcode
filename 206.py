# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
        def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
        	return head
        p = head
        d = {}
        i = 0
        while p:
        	d[i] = p
        	p = p.next
        	i += 1
        l = len(d)
        for i in range(l-1,0,-1):
        	d[i].next = d[i-1]
        d[0].next = None
        return d[l-1]




