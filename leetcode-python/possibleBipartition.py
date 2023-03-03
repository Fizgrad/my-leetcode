class Solution:
    def possibleBipartition(self, n: int, dislikes: list[list[int]]) -> bool:
        color = [0 for _ in range(n)]
        if len(dislikes) == 0:
            return True
        a, b = dislikes[0]
        color[a - 1] = 1
        color[b - 1] = 2
        visited = [0 for _ in range(len(dislikes))]
        visited[0] = 1

        while sum(visited) < len(visited):
            flag_no_change = True
            for i in range(1, len(dislikes)):
                if not visited[i]:
                    a, b = dislikes[i]
                    if color[a - 1]:
                        flag_no_change = False
                        visited[i] = 1
                        if color[b - 1]:
                            if color[a - 1] == color[b - 1]:
                                return False
                        else:
                            color[b - 1] = 3 - color[a - 1]
                    elif color[b - 1]:
                        flag_no_change = False
                        visited[i] = 1
                        color[a - 1] = 3 - color[b - 1]
            if flag_no_change:
                for i in range(n):
                    if not visited[i]:
                        idx = i
                        break
                    if i == n - 1:
                        return True
                a, b = dislikes[idx]
                color[a - 1] = 1
                color[b - 1] = 2
                visited[idx] = 1
        return True


if __name__ == '__main__':
    s = Solution()
    print(s.possibleBipartition(4, [[1, 2], [1, 3], [2, 4]]))
