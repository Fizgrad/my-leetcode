from bisect import bisect_left


class Solution:
    def jobScheduling(self, s, e, p):
        jobs, n = sorted(zip(s, e, p)), len(s)
        dp = [0] * (n + 1)
        for i in reversed(range(n)):
            k = bisect_left(jobs, jobs[i][1], key=lambda j: j[0])
            print(k)
            dp[i] = max(jobs[i][2] + dp[k], dp[i + 1])
            print(dp)

        return dp[0]


if __name__ == '__main__':
    a = Solution()
    print(a.jobScheduling(s=[1, 2, 3, 3], e=[3, 4, 5, 6], p=[50, 10, 40, 70]))
