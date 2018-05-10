class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        i = 0
        while i < len(prices)-1:
        	j = i+1
        	if prices[i] >= prices[j]:
        		i+=1
        	else:
        		while j <= len(prices)-1:
        			if j == len(prices)-1:
        				profit+=prices[j] - prices[i]
        				return profit
        			elif prices[j] <= prices[j+1]:
        				j+=1
        			else:
        				profit+=prices[j]-prices[i]
        				i = j+1
        				break
        return profit




