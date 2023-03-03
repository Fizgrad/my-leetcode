class Solution:
    def closestCost(self, baseCosts: list[int], toppingCosts: list[int], target: int) -> int:
        n1, n2, resList = len(baseCosts), len(toppingCosts), []
        for i in range(n1):
            costList = [baseCosts[i]]
            for j in range(n2):
                temp = []
                for k in costList:
                    oneAdd, twoAdd = k + toppingCosts[j], k + toppingCosts[j] + toppingCosts[j]
                    if k == target:
                        return target
                    if k < target:
                        temp.append(oneAdd)
                    if oneAdd < target:
                        temp.append(twoAdd)
                costList.extend(temp)
            minDif, res = abs(target - costList[0]), costList[0]
            for j in costList[1::]:
                dif = abs(target - j)
                if dif < minDif:
                    minDif, res = dif, j
                if dif == minDif and j < res:
                    res = j
            resList.append(res)
        minDif, res = abs(target - resList[0]), resList[0]
        for j in resList[1::]:
            dif = abs(target - j)
            if dif < minDif:
                minDif, res = dif, j
            if dif == minDif and j < res:
                res = j
        return res


if __name__ == '__main__':
    s = Solution()
    print(s.closestCost([3, 10], toppingCosts=[2, 5], target=9))
