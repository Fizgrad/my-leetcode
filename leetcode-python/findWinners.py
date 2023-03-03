from collections import defaultdict
from bisect import bisect_left


class Solution:
    def findWinners(self, matches: list[list[int]]) -> list[list[int]]:
        d, s, n = defaultdict(int), defaultdict(int), len(matches)
        for i in range(n):
            s[matches[i][0]] = 1
            s[matches[i][1]] = 1
            if d.get(matches[i][1]):
                d[matches[i][1]] += 1
            else:
                d[matches[i][1]] = 1
        res1, res2 = [], []
        for i in s:
            if d.get(i):
                if d[i] == 1:
                    k = bisect_left(res2, i)
                    res2.insert(k, i)
            else:
                k = bisect_left(res1, i)
                res1.insert(k, i)
        return [res1, res2]


if __name__ == '__main__':
    s = Solution()
    print(s.findWinners([[2, 3], [1, 3], [5, 4], [6, 4]]))
