class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        r = {}
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
        	return nums[0]
        if len(nums) == 2:
        	return max(nums[0],nums[1])
        else:
            r[0] = nums[0]
            r[1] = max(nums[0],nums[1])
            for i in range(2,len(nums)):
                r[i] = max(r[i-1],r[i-2]+nums[i])
        return r[len(nums)-1]



