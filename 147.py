# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        p = new_s = ListNode(None)
        new_s.next = head
        cur = head
        while cur and cur.next:
            val = cur.next.val
            if val > cur.val:
            	cur = cur.next
            	continue
            if p.next.val > val:
            	p = new_s
            while p.next.val < val:
            	p = p.next
            new = cur.next
            cur.next = new.next
            new.next = p.next
            p.next = new
        return new_s.next
        #second solution
        '''
        h = head
        res = []
        while h:
            res.append(h.val)
            h = h.next
        res.sort()
        h = head
        for i in res:
            h.val = i
            h = h.next
        return head

        '''














