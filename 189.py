# coding:utf8
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        #方法1
        '''def rev(start, end , s):
            while start < end:
                s[end] ,s[start]= s[start], s[end]
                end-=1
                start+=1
        length = len(nums)
        rev(length-k%length,length-1,nums)
        rev(0,length-k%length-1,nums)
        rev(0,length-1,nums)'''
        #方法2
        '''
        l = len(nums)
        nums[:] = nums[l-k:] + nums[:l-k]'''
        #方法3
        l = len(nums)
        nums[:l-k] = reversed(nums[:l-k])
        nums[l-k:] = reversed(nums[l-k:])
        nums[:] = reversed(nums)
        

