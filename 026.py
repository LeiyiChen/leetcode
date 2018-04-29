class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        l = len(nums)
        if l:
	        while i < l-1:
	        	if nums[i] == nums[i+1]:
	        		nums.pop(i+1)
	        		l-=1
	        		j = i+1
	        		while j < l:
	        			if nums[i] != nums[j]:
	        				break
	        			else:
	        				nums.pop(j)
	        				l-=1
	        	i+=1
        return len(nums)
s = Solution()
print(s.removeDuplicates([1,1,1]))


