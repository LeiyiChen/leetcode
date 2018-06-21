class Solution:
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        d = {}
        for i in nums:
        	if i in d:
        		d[i] += 1
        	else:
        		d[i] = 0
        l = sorted(list(d.keys()))
        if len(l) < 3:
        	return max(l)
        return l[-3]
        '''
        # better solution
        res = [float("-inf")] * 3
        for i in nums:
        	if i in res:
        		continue
        	if i > res[0]:
        		res = [i,res[0],res[1]]
        	elif i > res[1]:
        		res = [res[0],i,res[1]]
        	elif i > res[2]:
        		res = [res[0],res[1],i]
        	print(res)
        return res[-1] if res[2] != float("-inf") else res[0]




