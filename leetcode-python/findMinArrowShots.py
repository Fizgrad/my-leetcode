import math


class Solution:
    def findMinArrowShots(self, points: list[list[int]]) -> int:
        points = sorted(points, key=lambda x: x[0])
        count = 1
        left = math.inf
        for start, end in points:
            if end < left:
                left = end
            if left < start:
                count += 1
                left = end

        return count


if __name__ == '__main__':
    s = Solution()
    print(s.findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]))
