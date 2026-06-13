class Solution:
    def mapWordWeights(self, words: List[str], weights: List[int]) -> str:
        return "".join([chr(ord('a') + 25 - x) for x in [sum([weights[ord(c) - ord('a')] for c in word])%26 for word in words]])