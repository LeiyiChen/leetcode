class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        l = len(s)
        if l == 0:
            return 0
        n = l-1
        while n >= 0:
            if s[n] == ' ':
                n-=1
            else:
                for i in range(n-1,-1,-1):
                    if s[i] == ' ':
                        return n - i
                return n+1
        return 0

s = Solution()
print(s.lengthOfLastWord('ab '))
