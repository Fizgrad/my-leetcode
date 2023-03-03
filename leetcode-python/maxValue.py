class Solution:
    def maxValue(self, n, index, maxSum):
        maxSum -= n
        l, r = 0, maxSum
        while l != r:
            m = (l + r + 1) // 2
            left = min(index, m)
            right = min(m, n - index -1)
            if maxSum >= ((m*2-1 - left)*left + (m*2-1 -right)*right)/2 + m:
                l = m
            else:
                r = m - 1
        return r + 1


if __name__ == '__main__':
    s = Solution()
    print(s.maxValue(4, 0, 4))
    print(s.maxValue(4, 2, 6))
    print(s.maxValue(3, 0, 815094800))
