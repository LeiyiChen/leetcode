class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 1:
            return 1
        l = 1
        h = x//2
        while h > l:
            m = (l+h)//2
            if m**2 > x:
                h = m
            elif m**2 < x:
                l = m
            else:
                return m
        if (m-1)**2 < x < m**2:
        	return m-1
                
s = Solution()
print(s.mySqrt(8))