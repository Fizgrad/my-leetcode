import random


class RandomizedSet:
    def __init__(self):
        self.idxs = {}
        self.vals = []

    def insert(self, val: int) -> bool:
        if val in self.idxs:
            return False
        else:
            self.idxs[val] = len(self.vals)
            self.vals.append(val)
            return True

    def remove(self, val: int) -> bool:
        if val not in self.idxs:
            return False
        else:
            self.vals[self.idxs[val]], self.vals[len(self.vals) - 1] = self.vals[len(self.vals) - 1], self.vals[
                self.idxs[val]]
            self.idxs[self.vals[self.idxs[val]]] = self.idxs[val]
            self.vals.pop()
            self.idxs.pop(val)
            return True

    def getRandom(self) -> int:
        return random.choice(self.vals)

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
