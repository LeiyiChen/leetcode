class Solution:
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        d = {}
        for i in range(len(nums)):
        	if nums[i] in d:
        		if i - d[nums[i]] <= k:
        			return True
        		else:
        			d[nums[i]] = i
        	d[nums[i]] = i
        return False
