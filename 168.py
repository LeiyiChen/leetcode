class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        d = {}
        r = []
        a =''
        for i in range(1,27):
        	d[i] = chr(64+i)
        if n <= 26:
        	return d[n]
        if n % 26 == 0:
        	n = n/26 - 1
        	a ='Z'
        while n > 26:
        	s = n % 26
        	n = n // 26
        	r.append(s)
        result = d[n]
        for i in r[::-1]:
        	result+=d[i]
        return result + a






