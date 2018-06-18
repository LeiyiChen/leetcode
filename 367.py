class Solution:
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        l = 1
        h = num
        while l <= h:
        	mid = (l+h)//2
        	t = mid**2
        	if t < num:
        		l = mid + 1
        	elif t == num:
        		return True
        	else:
        		h = mid - 1
        return False


