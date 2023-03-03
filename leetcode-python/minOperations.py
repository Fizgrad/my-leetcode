class Solution:
    def minOperations(self, nums: list[int], x: int) -> int:
        left = 0
        sum_left = 0
        res = []
        if sum(nums) < x:
            return -1
        while sum_left < x:
            sum_left += nums[left]
            left += 1
            if sum_left == x:
                res.append(left)
        right = 0
        sum_right = 0
        while sum_right < x:
            sum_right += nums[-(right+1)]
            right += 1
            while left > 0 and sum_right + sum_left > x:
                left -= 1
                sum_left -= nums[left]
            if sum_left + sum_right == x:
                res.append(left + right)
        return min(res) if len(res) else -1


if __name__ == '__main__':
    s = Solution()
    print(s.minOperations([1, 1, 4, 2, 3], 5))
    print(s.minOperations([1, 1, ], 3))
