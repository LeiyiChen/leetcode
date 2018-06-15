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
        		del nums[i-k]
        		nums.append(0)
        		k += 1
        '''
        #second solution
        '''
        n = nums.count(0)
        for i in range(n):
            nums.remove(0)
        nums.extend([0]*n)
        '''
        # third solution
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j], nums[i] = nums[i], nums[j]
                j += 1

