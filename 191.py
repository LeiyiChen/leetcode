class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = 0
        l = list(map(int,list('{:b}'.format(n))))
        for i in l:
        	r += i
        return r


