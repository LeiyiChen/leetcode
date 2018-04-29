class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        l = digits[::]
        i = len(digits)-1
        while i >= 0:
            if l[i]+1 <=9:
                l[i]+=1
                return l
            else:           
                if i == 0:
                    l[i] = (l[i]+1)%10
                    l.insert(0,1)
                    return l        
                elif l[i]+1 > 9:
                    l[i]=(l[i]+1)%10
                    if l[i-1]+1 < 10:
                        l[i-1] = l[i-1]+1
                        return l
                    else:
                        i-=1

s = Solution()
print(s.plusOne([1,2,1]))