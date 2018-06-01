class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        d = {}
        while True:
        	m = 0
        	while n > 0:
        		m += (n%10)**2
        		print(m)
        		n //= 10 
        	if m in d:
        		print(d)
        		return False
        	if m == 1:
        		print(d)
        		return True
        	d[m] = m
        	n = m
s = Solution()
print(s.isHappy(2))
