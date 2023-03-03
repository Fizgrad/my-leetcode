class Solution:
    def uniqueOccurrences(self, arr: list[int]) -> bool:
        sets = {}
        for i in arr:
            if i in sets:
                sets[i] += 1
            else:
                sets[i] = 1
        nums = {}
        for i in sets:
            if sets[i] in nums:
                return False
            else:
                nums[sets[i]] = 1
        return True


if __name__ == '__main__':
    s = Solution()
    print(s.uniqueOccurrences([1, 2, 2, 1, 1, 3]))
