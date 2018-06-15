class Solution:
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        while n > 1 :
	        l = list(map(int, str(n)))
	        if sum(l) % 3 == 0:
	        	n = n // 3
	        else:
	        	return False
        if n <= 0:
        	return False
        return True
s =Solution()
print(s.isPowerOfThree())
        