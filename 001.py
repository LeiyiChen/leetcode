class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dic = dict()
        for index,value in enumerate(nums):
            sub = target - value
            if sub in dic:#只对字典的键进行查找
                return [dic[sub],index]
            else:
                dic[value] = index