import math


class Solution:
    def nearestValidPoint(self, x: int, y: int, points: list[list[int]]) -> int:
        minDis, idx, n = math.inf, -1, len(points)
        for i in range(n):
            disX, disY = abs(x - points[i][0]), abs(y - points[i][1])
            if disX == 0 or disY == 0:
                if disX + disY < minDis:
                    minDis, idx = dis, i
        return idx
