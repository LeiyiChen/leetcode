class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        l = len(nums)
        if l == 0:
        	return 0
        i = 0
        while i < l:
	       	if nums[i] == val:
	       		nums.pop(i)
	       		l-=1
	       	else:
	       		i+=1
       	return len(nums)
if __name__ == '__main__':
	s = Solution()
	print(s.removeElement([1,2],3))