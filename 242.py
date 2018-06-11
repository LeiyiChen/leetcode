class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s = sorted(list(s))
        l = sorted(list(l))
        return s == l
        # second solution
        return set(s) == set(t) and all(s.count(i) == t.count(i) for i in set(s))
        # third solution
        for n in 'abcdefghijklmnopqrstuvwxyz':
            if s.count(n) != t.count(n):
                return False
        return True
                