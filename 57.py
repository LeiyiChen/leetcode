# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        intervals.append(newInterval)
        l = len(intervals)
        res = []
        intervals = sorted(intervals, key = lambda intervals:intervals.start)
        low = intervals[0].start
        high = intervals[0].end
        for i in range(1, l):
        	if intervals[i].start <= high:
        		high = max(high, intervals[i].end)
        	else:
        		res.append([low, high])
        		low = intervals[i].start
        		high = intervals[i].end
        res.append([low, high])
        return res








