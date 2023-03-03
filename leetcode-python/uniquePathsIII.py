import functools


class Solution:
    def uniquePathsIII(self, grid: list[list[int]]) -> int:
        visited = [[not x == 0 for x in i] for i in grid]
        count = [0]
        if (m := len(grid) )> 0:
            n = len(grid[0])
        d = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        isEnd = lambda visited: functools.reduce(lambda a, b: a and b,
                                                 [functools.reduce(lambda a, b: a and b, i) for i in visited])
        def dfs(x, y, visited, grid, count):
            for dx, dy in d:
                xx = x + dx
                yy = y + dy
                if 0 <= xx < m and 0 <= yy < n:
                    if grid[xx][yy] == 2:
                        if isEnd(visited):
                            count[0] += 1
                    elif not visited[xx][yy]:
                        if grid[xx][yy] == 0:
                            visited[xx][yy] = True
                            dfs(xx, yy, visited=visited, grid=grid, count=count)
                            visited[xx][yy] = False
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(i, j, visited=visited, grid=grid, count=count)
                    return count[0]


if __name__ == '__main__':
    s = Solution()
    print(s.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, -1]]))
