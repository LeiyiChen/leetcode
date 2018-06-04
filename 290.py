class Solution:
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        l = str.split(' ')
        if len(l) != len(pattern):
        	return False
        return len(set(zip(pattern,l))) == len(set(pattern)) == len(set(l))
