class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        buff = 0
        s = ''
        flag = 1
        flag1 = 1
        l = min(len(a), len(b))
        if len(a) > len(b):
            temp = a[len(a) - l:]
            min_string = b
            other = a[:len(a) - l]
        elif len(a) < len(b):
            temp = b[len(b) - l:]
            min_string = a
            other = b[:len(b) - l]
        else:
            flag1 = 0
            temp = a
            min_string = b
        for i in range(l - 1, -1, -1):
            if buff == 0:
                if temp[i] == min_string[i] == '0':
                    s += '0'
                    buff = 0
                elif temp[i] == min_string[i] == '1':
                    s += '0'
                    buff = 1
                else:
                    s += '1'
                    buff = 0
            else:
                if temp[i] == min_string[i] == '0':
                    s += '1'
                    buff = 0
                elif temp[i] == min_string[i] == '1':
                    s += '1'
                    buff = 1
                else:
                    s += '0'
                    buff = 1
        if not flag1:
            if buff == 1:
                s += '1'
        elif not buff:
            s += other[::-1]
        else:
            for j in range(len(other) - 1, -1, -1):
                if other[j] == '0':
                    s += '1'
                    if j == 1:
                        s+=other[:1]
                    else:
                        other = other[:j]
                        s += other[::-1]
                    flag = 0
                    break
                else:
                    s += '0'
                    if j == 0:
                        s += '1'
        return s[::-1]
a = Solution()
print(a.addBinary('1010','1011'))

