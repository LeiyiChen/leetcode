class Solution:
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = ['a','o','e','u','i','A','O','E','U','I']
        res = list(s)
        index_list = []
        for i in range(len(res)):
            if res[i] in l:
                index_list.append(i)
        length = len(index_list)
        for j in range(length//2):
            res[index_list[j]], res[index_list[-j-1]] = res[index_list[-j-1]], res[index_list[j]]
        print(index_list)
        return ''.join(res)

