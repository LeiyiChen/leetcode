class Solution:
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # first solution
        '''
        i = 0
        nums.sort()
        for item in nums:
        	if item - i != 0:
        		return i
        	else:
        		i += 1
        return i
        '''
        #second solution
        l = len(nums)
        s = l*(l+1)//2
        return s - sum(nums)
