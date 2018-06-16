class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = list(s)[::-1]
        res = ''.join(l)
        return res