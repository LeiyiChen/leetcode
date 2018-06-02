# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if head:
            while head.val == val:
                head = head.next
                if head is None:
                    return head
            q = head
            p = q.next
            while p:
                if p.val == val:
                    q.next = p.next
                else:
                    q = q.next
                p = p.next
        return head
        #second solution
        '''
        pre_node = ListNode(None)
        pre_node.next = head
        q = pre_node
        while q.next:
            if q.next.val == val:
                q.next = q.next.next
            else:
                q = q.next
        '''

