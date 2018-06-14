# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        l = 1
        h = n
        while l <= h:
        	m = (l+h)//2
        	if isBadVersion(m) and isBadVersion(m-1):
        		h = m-1
        	elif isBadVersion(m):
        		return m
        	elif isBadVersion(m+1):
        		return m + 1
        	else:
        		l = m + 1
        return m