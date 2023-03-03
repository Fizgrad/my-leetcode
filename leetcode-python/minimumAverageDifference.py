import math


class Solution:
    def minimumAverageDifference(self, nums: list[int]) -> int:
        sumOfNums, n = sum(nums), len(nums)
        minAD, minIdx = math.inf, -1
        sumL, sumR = 0, sumOfNums
        for i in range(n):
            sumL += nums[i]
            sumR -= nums[i]
            nR = (n - i - 1)
            if nR == 0:
                AD = sumL / (i + 1)
            else :
                AD = abs(math.floor(sumL / (i + 1)) - math.floor(sumR /nR ))

            if AD < minAD:
                minIdx = i
                minAD = AD
        return minIdx


if __name__ == '__main__':
    s = Solution()
    print(s.minimumAverageDifference([0,1,0,1,0,1]))
