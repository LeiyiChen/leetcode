class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s = {}
        for i in nums:
        	if i in s.keys():
        		s.pop(i)
        	else:
        		s[i]=1
        return list(s.keys())[0]
        '''another solution
        res = 0
        for i in nums:
            res^=i
        return res
        '''
