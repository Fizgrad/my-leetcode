class Solution:
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        s, n, res = [], len(temperatures), temperatures.copy()
        for i in range(n):
            while len(s) and temperatures[s[-1]] < temperatures[i]:
                res[s[-1]] = i - s[-1]
                s.pop()
            s.append(i)
        while len(s):
            res[s[-1]] = 0
            s.pop()
        return res


if __name__ == '__main__':
    s = Solution()
    print(s.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
