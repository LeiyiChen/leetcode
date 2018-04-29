# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        p = head
        if p.next == None:
            return head
        try:
            while p.next:
                i = p.val
                if p.next.val == i:
                    p.next = p.next.next
                    while p.next:
                        if p.next.val != i:
                            break
                        p.next = p.next.next
                p = p.next
        except AttributeError:
            return head
        else:
            return head


