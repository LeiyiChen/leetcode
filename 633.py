class Solution:
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        l = 0
        h = int(c**0.5)
        while l <= h:
        	tmp = l ** 2 + h ** 2
        	if tmp < c:
        		l += 1
        	elif tmp == c:
        		return True
        	else:
        		h -= 1
        return False

