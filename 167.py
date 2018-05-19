class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        s = []
        r = []
        for i in range(len(numbers)):
        	if numbers[i] in s:
        		r.append(s.index(numbers[i]))
        		r.append(i)
        		return r
        	s.append(target-numbers[i])
        return None

if __name__=="__main__":
	s = Solution()
	print(s.twoSum([1,7,11,15],9))
