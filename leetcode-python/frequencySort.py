class Solution:
    def frequencySort(self, s: str) -> str:
        freq, cha = {}, []
        for i in s:
            if i not in freq:
                freq[i] = 1
                cha.append(i)
            else:
                freq[i] += 1
        cmp = lambda a: freq[a]
        cha.sort(reverse=True, key=cmp)
        res = ""
        for i in cha:
            for j in range(freq[i]):
                res += i
        return res


if __name__ == '__main__':
    s = Solution()
    print(s.frequencySort("Aabb"))
