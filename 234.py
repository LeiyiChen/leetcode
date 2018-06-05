# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
        	return True
        fast = slow = head
        while fast and fast.next:
        	fast = fast.next.next
        	slow = slow.next
        def reverse_list(head):
        	if head is None:
        		return head
        	cur = head
        	pre = None
        	nxt = cur.next
        	while nxt:
        		cur.nxt = pre
        		pre = cur
        		cur = nxt
        		nxt = nxt.next
        	cur.next = pre
        	return cur
        	p = reverse_list(slow.next)
        	while p.next:
        		if p.val != head.val:
        			return False
        		p = p.next
        		slow = slow.next
        	return True
        #second solution
        '''
        if head is None or head.next is None:
        	return True
        if head.next.next is None:
            return head.val == head.next.val
        fast = slow = q = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        def reverse_list(head):
        	if head is None:
        		return head
        	cur = head
        	pre = None
        	nxt = cur.next
        	while nxt:
        		cur.next = pre
        		pre = cur
        		cur = nxt
        		nxt = nxt.next
        	cur.next = pre
        	return cur
        p = reverse_list(slow.next)
        while p.next:
        	if p.val != q.val:
        		return False
        	p = p.next
        	q = q.next
        return p.val == q.val
        '''


