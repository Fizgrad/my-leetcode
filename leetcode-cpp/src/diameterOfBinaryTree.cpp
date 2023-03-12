//
// Created by David Chen on 3/12/23.
//
/**
 * Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    int res = 0;
    int diameterOfBinaryTreeDFS(TreeNode *root) {
        if (root == nullptr) {
            return 0;
        }
        int left = diameterOfBinaryTreeDFS(root->left);
        int right = diameterOfBinaryTreeDFS(root->right);
        res < left + right ? (res = left + right) : res;
        return (left > right ? left : right) + 1;
    }

    int diameterOfBinaryTree(TreeNode *root) {
        diameterOfBinaryTreeDFS(root);
        return res;
    }
};