class Solution:
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) == 0 or len(nums) == 1:
        	return False
        d = {}
        for i in nums:
        	if i in d:
        		return True
        	d[i] = 0
        return False
      