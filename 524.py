class Solution:
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        d = {}
        for i in s:
        	if i not in d:
        		d[i] = 1
        	else:
        		d[i] += 1
        res = sorted(d, key = lambda x: len(x))
        for item in res[::-1]:
        	i = 0
        	while i < len(item):
        		if item[i] in d and item.count(item[i]) <= d[i]:
        			i += 1
        		else: 
        			break
        	if i == len(item):
        		return item
        return None
s = Solution()
print(s.findLongestWord("abpcplea",["ale","apple","monkey","plea"])) 


