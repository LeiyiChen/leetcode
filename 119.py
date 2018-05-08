class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        p = [1]
        if not rowIndex:
            return p
        for i in range(rowIndex):
            s = list(map(lambda x,y:x+y, [0]+p,p+[0]))
            p = s
        return s
