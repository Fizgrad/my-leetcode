class Solution:
    def sumOfDistancesInTree(self, N, edges):
        dic1 = collections.defaultdict(list)
        for e in edges:
            dic1[e[0]].append(e[1])
            dic1[e[1]].append(e[0])

        exclude = {0}

        # eachItem subtreeDist[n]=[a, b] means subtree rooted at n has totally a nodes,
        # and sum of distance in the subtree for n is b
        subtreeDist = [[0, 0] for _ in range(N)]

        ans = [0] * N

        def sumSubtreeDist(n, exclude):
            cnt, ret = 0, 0
            exclude.add(n)
            for x in dic1[n]:
                if x in exclude:
                    continue
                res = sumSubtreeDist(x, exclude)
                cnt += res[0]
                ret += (res[0] + res[1])
            subtreeDist[n][0] = cnt + 1
            subtreeDist[n][1] = ret
            return cnt + 1, ret

        # recursively calculate the sumDist for all subtrees
        # 0 can be replaced with any other number in [0, N-1]
        # and the chosen root has its correct sum distance in the whole tree
        sumSubtreeDist(0, set())

        # visit and calculates the sum distance in the whole tree
        def visit(n, pre, exclude):
            if pre == -1:
                ans[n] = subtreeDist[n][1]
            else:
                ans[n] = ans[pre] - 2 * subtreeDist[n][0] + N
            exclude.add(n)
            for x in dic1[n]:
                if x not in exclude:
                    visit(x, n, exclude)

        visit(0, -1, set())
        return ans

if __name__ == '__main__':
    s = Solution()
    print(s.sumOfDistancesInTree(100, [[74, 34], [67, 44], [81, 40], [1, 97], [44, 88], [95, 23], [77, 78], [67, 29],
                                       [98, 1], [89, 3], [60, 91], [30, 28], [64, 85], [47, 72], [64, 9], [26, 35],
                                       [24, 1], [43, 35], [62, 86], [92, 86], [59, 89], [31, 3], [31, 92], [1, 33],
                                       [54, 68], [57, 63], [2, 3], [36, 64], [6, 9], [3, 67], [99, 70], [9, 47],
                                       [45, 16], [94, 92], [22, 9], [56, 31], [89, 84], [40, 31], [37, 38], [57, 52],
                                       [75, 76], [1, 26], [65, 79], [5, 39], [96, 47], [55, 14], [83, 54], [6, 32],
                                       [11, 26], [8, 40], [32, 69], [32, 14], [78, 79], [34, 92], [31, 75], [39, 45],
                                       [3, 79], [71, 31], [82, 74], [51, 58], [27, 35], [60, 70], [31, 51], [53, 74],
                                       [64, 60], [84, 90], [39, 40], [28, 80], [0, 47], [31, 41], [1, 25], [56, 48],
                                       [93, 10], [1, 17], [37, 7], [47, 15], [49, 41], [5, 18], [4, 92], [25, 64],
                                       [84, 95], [10, 95], [63, 66], [46, 87], [92, 50], [66, 3], [64, 75], [61, 98],
                                       [78, 12], [54, 71], [7, 65], [87, 39], [73, 96], [61, 20], [64, 19], [21, 69],
                                       [30, 6], [42, 72], [13, 67]]))
