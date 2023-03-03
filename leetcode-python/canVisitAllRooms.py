class Solution:
    def canVisitAllRooms(self, rooms: list[list[int]]) -> bool:
        n = len(rooms)
        s = set()
        visited = [False for _ in range(n)]
        visited[0] = True
        for i in rooms[0]:
            s.add(i)
            visited[i] = True
        while len(s):
            idx = s.pop()
            for i in rooms[idx]:
                if not visited[i]:
                    s.add(i)
                    visited[i] = True
        for i in visited:
            if not i:
                return False
        return True


if __name__ == '__main__':
    s = Solution()
    print(s.canVisitAllRooms([[1],[2],[3],[]]))