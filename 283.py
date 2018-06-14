class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        #first solution
        '''
        k = 0
        for i in range(len(nums)):
        	if nums[i-k] == 0:
        		del nums[i]
        		nums.append(0)
        		k += 1
        '''
        #second solution
        '''
        k = 0
        for i in range(len(nums)):
        	if nums[i-k] == 0:
        		del nums[i]
        		nums.append(0)
        		k += 1
        '''
        # third solution
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j], nums[i] = nums[i], nums[j]
                j += 1

