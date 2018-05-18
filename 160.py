# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a = headA
        b = headB
        i = 0
        l = m =0
        while a:
            a = a.next
            l += 1
        while b:
            b = b.next
            m += 1
        if a != b:
            return None
        a = headA
        b = headB        
        if l > m:
            diff = l - m
            while diff:
                a = a.next
                diff-=1
        if l < m:
            diff = m - l
            while diff:
                b = b.next
                diff-=1
        while a!=b:
            a = a.next
            b = b.next
        return a