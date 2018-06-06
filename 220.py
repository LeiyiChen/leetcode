class Solution:
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        d = {}
        for i in range(len(nums)):
        	d1 = d
        	for key in d1:
        		if abs(nums[i]-key) <= t:
        			if -k <= i - d1[key] <= k:
        				return True
        		d[key] = i
        return False
s = Solution()
print(s.containsNearbyAlmostDuplicate([1,2,3,1],3,0))

