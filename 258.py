class Solution:
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        s = str(num)
        res = 0
        for i in s:
        	res += int(i)
        while res > 9:
        	s = str(res)
        	res = 0
        	for i in s:
        		res += int(i)
        return res
        #second solution
        return num and (num % 9 or 9) 

