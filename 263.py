class Solution:
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        l = [2,3,5]
        if num < 1:
        	return False
        if num == 1 or num in l:
        	return True
        i = 0
        while i < len(l):
        	if num % l[i] != 0:
        		i += 1
        	else:
        		num = num // l[i]
        		if num in l:
        			return True
        		i = 0
        return False
