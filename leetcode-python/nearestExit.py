class Solution:
    def nearestExit(self, maze: list[list[str]], entrance: list[int]) -> int:
        x = len(maze)
        y = len(maze[0])
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        checked = []
        for i in maze:
            temp_list = []
            for _ in i:
                temp_list.append(False)
            checked.append(temp_list)
        entrance.append(0)
        queue = [entrance]
        while len(queue) > 0:
            front = queue[0]
            if (not (front[0] == entrance[0] and front[1] == entrance[1])) and (
                    front[0] == 0 or front[0] == x - 1 or front[1] == 0 or front[1] == y - 1):
                return front[2]
            for i in range(4):
                xx = front[0] + dx[i]
                yy = front[1] + dy[i]
                if 0 <= xx < x and 0 <= yy < y:
                    if not checked[xx][yy] and maze[xx][yy] == '.':
                        queue.append([xx, yy, front[2] + 1])
                        checked[xx][yy] = True
            queue.pop(0)
        return -1


if __name__ == "__main__":
    s = Solution()
    print(s.nearestExit([[".", "."]], [0, 1]))
