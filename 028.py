class Solution:
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle  not in haystack:
        	return -1
        return haystack.find(needle)

s = Solution()
print(s.strStr('sadfrr','ad'))
print(s.strStr('a','b'))
