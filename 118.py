class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if not numRows:
        	return []
        p = [1]
        r = [p]
        i = 1
        while i < numRows:
        	i+=1
        	s = [0] * i
        	p = [0] + p + [0]
        	for j in range(i):
        		s[j] = p[j] + p[j+1]
        	r.append(s)
        	p = s 
        return r

