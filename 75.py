class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        index = 0
        for i in range(1, l):
            if nums[i] < nums[index]:
                index = i
        nums[0], nums[index] = nums[index], nums[0]
        k = j = 1
        while j < l:
            if nums[j] == 2:
                nums.pop(j)
                nums.append(2)
                l -= 1
            elif nums[j] == 0:
                nums[k], nums[j] = nums[j], nums[k]
                k += 1
                j += 1
            else:
                j += 1
        return nums

