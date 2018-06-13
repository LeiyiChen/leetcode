class Solution:
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 0:
            return False
        t1 = 0
        t2 = 0
        t3 = 0
        res = [1]
        while len(res) < n:
        	res.append(min(res[t1]*2, res[t2]*3, res[t3]*5))
	        if res[-1] == res[t1]*2:
	        	t1 += 1
	        if res[-1] == res[t2]*3:
	        	t2 += 1
	        if res[-1] == res[t3]*5:
	        	t3 += 1
        return res[-1]