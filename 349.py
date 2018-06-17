class Solution:
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        l1 = len(nums1)
        l2 = len(nums2)
        res = set()
        if l1 <= l2:
            s = nums1
            l = nums2
        else:
            l = nums1
            s = nums2
        for i in l:
            if i in s:
                res.add(i)
        return list(res)
