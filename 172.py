class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = 0
        while n >= 5:
            n = n // 5
            r+=n
        return r 
 	

