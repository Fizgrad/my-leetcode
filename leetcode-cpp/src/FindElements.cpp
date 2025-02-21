#include <unordered_set>


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class FindElements {
public:
    std::unordered_set<int> sets;

    FindElements(TreeNode *root) {
        auto dfs = [&](auto &&dfs, TreeNode *node, int val) -> void {
            if (node == nullptr) return;
            sets.insert(val);
            if (node->left)
                dfs(dfs, node->left, val * 2 + 1);
            if (node->right)
                dfs(dfs, node->right, val * 2 + 2);
            return;
        };
        dfs(dfs, root, 0);
    }

    bool find(int target) {
        return sets.contains(target);
    }
};

/**
     * Your FindElements object will be instantiated and called as such:
     * FindElements* obj = new FindElements(root);
     * bool param_1 = obj->find(target);
     */

int main() {
    return 0;
}