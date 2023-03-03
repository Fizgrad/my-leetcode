# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def procedure(self, node, res):
        if not node:
            return 0
        else:
            r = self.procedure(node.right, res)
            l = self.procedure(node.left, res)
            res[0] = max(res[0], r + l + node.val, node.val, l + node.val, r + node.val)
            return max(r, l, 0) + node.val

    def maxPathSum(self, root) -> int:
        res = [-100000000000000]
        self.procedure(root, res)
        return res[0]
