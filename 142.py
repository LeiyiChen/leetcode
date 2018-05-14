# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
        	return None
        fast = head
        slow = head
        while fast.next and fast.next.next:
        	slow = slow.next
        	i+=1
        	fast = fast.next.next
        	if slow == fast:
        		p = head
        		while slow != p:
        			p = p.next
        			slow = slow.next
        		return p
        return None
