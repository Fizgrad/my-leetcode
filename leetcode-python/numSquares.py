import math

# Legendre's three-square theorem
class Solution:
    def numSquares(self, n: int) -> int:
        if int(math.sqrt(n)) ** 2 == n:
            return 1

        for i in range(1, int(math.sqrt(n)) + 1):  # [2] check pairs of squares
            if (j := n - i ** 2) == int(math.sqrt(j)) ** 2:
                return 2

        temp = n
        while temp % 4 == 0:
            temp = temp / 4
        if temp % 8 != 7:
            return 3

        return 4


if __name__ == "__main__":
    s = Solution()
    print(s.numSquares(13))
