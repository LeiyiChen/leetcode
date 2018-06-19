class Solution:
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        #解题思路
        """
        利用&求进位，^异或求值
        但是在Python中并不可行，因为Python会直接将
        int扩展为long
        """
        # while b!=0:
        #     carry=a&b
        #     a=a^b
        #     b=carry<<1
        # return a
        while b != 0:
            carry = a & b
            a = (a ^ b) % 0x100000000
            b = (carry << 1) % 0x100000000
        return a if a <= 0x7FFFFFFF else a | (~0x100000000+1)