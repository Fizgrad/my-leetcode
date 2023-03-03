from collections import Counter


class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        res = 0
        for _, freq in Counter(tasks).items():
            if freq == 1:
                return -1
            res += (freq + 2) // 3
        return res


if __name__ == '__main__':
    s = Solution()
    print(s.minimumRounds([2, 2, 3, 3, 2, 4, 4, 4, 4, 4]))
