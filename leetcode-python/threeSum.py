def threeSum(nums: list[int]):
    res, n, nums, i = set(), len(nums), sorted(nums), 0
    while i < n:
        if i > 0 and nums[i] == nums[i - 1]:
            i += 1
            continue
        j, k = i + 1, n - 1
        while j < k:
            _sum = nums[i] + nums[j] + nums[k]
            if _sum == 0:
                res.add(tuple([nums[i], nums[j], nums[k]]))
                j, k = j + 1, k - 1
            elif _sum < 0:
                j += 1
            else:
                k -= 1
        i += 1
    return res


if __name__ == '__main__':
    print(threeSum([1, 2, -2, -1]))
