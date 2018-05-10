class Solution:
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
        	return True
        def f(i):
        	q = i.isalpha()
        	if 48<=ord(i)<=57:
        		p = 1
        	else:
        		p = 0
        	return q or p
        l = list((filter(f, s)))
        s = ''.join(l)
        s = s.upper()
        return s[::] == s[::-1]

