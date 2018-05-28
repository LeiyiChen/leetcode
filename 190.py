class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
    	l = list('{0:032b}'.format(n))
    	l = list(map(int,l))
    	l.reverse()
    	l = list(map(str,l))
    	s = ''.join(l)
    	return int(s,2)
s = Solution()
print(s.reverseBits(0))
