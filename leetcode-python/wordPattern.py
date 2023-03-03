class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        dict_pattern = dict()
        words = s.split(" ")
        idx = 0
        if not len(pattern) == len(words):
            return False
        for i in pattern:
            if i in dict_pattern:
                if not dict_pattern[i] == words[idx]:
                    return False
            else:
                if words[idx] in dict_pattern.values():
                    return False
                dict_pattern[i] = words[idx]
            idx += 1
        return True


if __name__ == '__main__':
    s = Solution()
    print(s.wordPattern("abba", "cat cat cat cat"))
    print(s.wordPattern("aaa", "aa aa aa aa"))
