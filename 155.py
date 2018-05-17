class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.l = []

        
    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if x is None:
            pass
        else:
            self.l.append(x)
        

    def pop(self):
        """
        :rtype: void
        """
        if self.l is None:
            return 'error'
        else:
            self.l.pop(-1)
        

    def top(self):
        """
        :rtype: int
        """
        if self.l is None:
            return 'error'
        else:
            return self.l[-1]
        

    def getMin(self):
        """
        :rtype: int
        """
        if self.l is None:
            return 'error'
        else:
            return min(self.l)