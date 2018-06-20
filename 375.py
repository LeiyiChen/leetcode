class Solution:
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(2, n+1):
            for j in range(i-1, 0, -1):
                global_min = float("inf")
                for k in range(j, i):
                    local_max = k + max(dp[j][k-1], dp[k+1][i])
                    global_min = min(local_max, global_min)
                dp[j][i] = global_min
        return dp[1][n]



    
        	