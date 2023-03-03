class Solution:
    def largestSumOfAverages(self, nums: list[int], k: int) -> float:
        n = len(nums)
        dp = [[0 for _ in range(k)].copy() for _ in range(n)]
        for i in range(n):
            dp[i][0] = (dp[i - 1][0] * i + nums[i]) / (i + 1)
            for j in range(1, min(i + 1, k)):
                dp[i][j] = max([dp[m][j - 1] + sum(nums[m + 1:i + 1]) / (i - m) for m in range(j - 1, i)])
        return dp[n - 1][k - 1]


if __name__ == '__main__':
    s = Solution()
    print(s.largestSumOfAverages([1, 2, 3], 3))
