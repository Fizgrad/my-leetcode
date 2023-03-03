class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        if (n := len(prices)) <= 1:
            return 0
        if n == 2:
            return max(0, prices[1] - prices[0])
        dp = [0, max(0, prices[1] - prices[0]),
              max(0, prices[1] - prices[0], prices[2] - prices[0], prices[2] - prices[1])]

        for i in range(2, n):
            dp.append(max([max(dp[x] + (prices[i] - prices[x + 2]),prices[i] - prices[x]) \
                           for x in range(0, i - 1)]))
            print(dp)
        return dp [-1]


if __name__ == '__main__':
    s = Solution()
    print(s.maxProfit([6, 1, 3, 2, 4, 7]))
