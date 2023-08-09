//
// Created by David Chen on 8/2/23.
//
#include<iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <array>
#include <set>
#include <stack>
#include <deque>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <climits>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(nullptr) {}
};

struct TrieNode {
    TrieNode *next['z' - 'a' + 1] = {nullptr};
    bool isEnd = false;
};

struct Trie {
    TrieNode *root = new TrieNode();

    bool contains(const string &input) {
        auto temp = root;
        for (int i = 0; i < input.size(); ++i) {
            temp = temp->next[input[i] - 'a'];
            if (temp == nullptr) {
                return false;
            }
        }
        return temp->isEnd;
    }

    void add(const string &input) {
        auto temp = root;
        for (int i = 0; i < input.size(); ++i) {
            if (temp->next[input[i] - 'a'] == nullptr)
                temp->next[input[i] - 'a'] = new TrieNode();
            temp = temp->next[input[i] - 'a'];
        }
        temp->isEnd = true;
    }
};


class Solution {
public:
    vector<string> letterCombinations(const string &digits) {
        if (digits.empty()) {
            return {};
        }
        unordered_map<char, string> chars{{'1', ""},
                                          {'2', "abc"},
                                          {'3', "def"},
                                          {'4', "ghi"},
                                          {'5', "jkl"},
                                          {'6', "mno"},
                                          {'7', "pqrs"},
                                          {'8', "tuv"},
                                          {'9', "wxyz"},
                                          {'0', ""}};
        vector<string> next = {""};
        vector<string> temp;
        for (auto i: digits) {
            for (auto c: chars[i]) {
                for (auto k: next) {
                    k.push_back(c);
                    temp.emplace_back(k);
                }
            }
            next.swap(temp);
            temp.clear();
        }
        return next;
    }

    bool wordBreak(const string &s, vector<string> &wordDict) {
        Trie trie;
        for (auto &i: wordDict) {
            trie.add(i);
        }
        int n = s.size();
        vector<vector<int >> dp(n, vector<int>(n, -1));
        auto f = [&](auto &&f, int i, int j) {
            if (dp[i][j] != -1) {
                return dp[i][j];
            } else {
                if (trie.contains(s.substr(i, j - i + 1))) {
                    return dp[i][j] = 1;
                }
                for (int k = 1; k <= j - i; ++k) {
                    if (trie.contains(s.substr(i, k)) && f(f, i + k, j)) {
                        return dp[i][j] = 1;
                    }
                }
                return dp[i][j] = 0;
            }
        };
        return f(f, 0, s.size() - 1);
    }

    vector<TreeNode *> generateTrees(int n) {
        vector<vector<vector<TreeNode *>>> dp(n + 1, vector<vector<TreeNode *>>(n + 1, vector<TreeNode *>()));
        auto f = [&](auto &&f, int low, int high) -> vector<TreeNode *> {
            if (low > high) {
                return {};
            }
            if (low == high) {
                return {new TreeNode(low)};
            }
            if (!dp[low][high].empty())
                return dp[low][high];
            else {
                vector<TreeNode *> res;
                for (int k = low; k <= high; ++k) {
                    auto left = f(f, low, k - 1);
                    auto right = f(f, k + 1, high);
                    if (left.empty()) {
                        for (auto j: right) {
                            res.emplace_back(new TreeNode(k, nullptr, j));
                        }
                    }
                    if (right.empty()) {
                        for (auto i: left) {
                            res.emplace_back(new TreeNode(k, i, nullptr));
                        }
                    }
                    for (auto i: left) {
                        for (auto j: right) {
                            res.emplace_back(new TreeNode(k, i, j));
                        }
                    }
                }
                return dp[low][high] = res;
            }
        };
        return f(f, 1, n);
    }

    bool searchMatrix(vector<vector<int>> &matrix, int target) {
        int n = matrix.size();
        int m = matrix.begin()->size();
        int low = 0;
        int high = n - 1;
        int i_index = 0;
        int mid = (low + high) >> 1;
        while (low <= high) {
            mid = (low + high) >> 1;
            if (matrix[mid][0] > target) {
                high = mid - 1;
            } else if (matrix[mid][0] == target) {
                return true;
            } else if (matrix[mid][0] < target) {
                i_index = mid;
                low = mid + 1;
            }
        }
        low = 0;
        high = m - 1;
        while (low <= high) {
            mid = (low + high) >> 1;
            if (matrix[i_index][mid] > target) {
                high = mid - 1;
            } else if (matrix[i_index][mid] == target) {
                return true;
            } else if (matrix[i_index][mid] < target) {
                low = mid + 1;
            }
        }
        return false;
    }

    int numMusicPlaylists(int n, int goal, int k) {
        using ll = long long;
        const int MOD = 1e9 + 7;
        vector<vector<int>> dp(n + 1, vector<int>(goal + 1, -1));
        auto solve = [&](auto &&solve, int n, int goal, int k) -> ll {
            if (n == 0 && goal == 0) return 1;
            if (n == 0 || goal == 0) return 0;
            if (dp[n][goal] != -1) return dp[n][goal];
            ll pick = solve(solve, n - 1, goal - 1, k) * n;
            ll notpick = solve(solve, n, goal - 1, k) * max(n - k, 0);
            return dp[n][goal] = (pick + notpick) % MOD;
        };
        return solve(solve, n, goal, k);
    }

    int search(vector<int> &nums, int target) {
        int n = nums.size();
        int low = 0;
        int high = n - 1;
        int mid;
        int cut;
        while (low <= high) {
            mid = (low + high) >> 1;
            if (nums[mid] >= nums[0]) {
                cut = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        if (target == nums[0]) {
            return 0;
        } else if (target > nums[0]) {
            low = 0;
            high = cut;
        } else {
            low = cut + 1;
            high = n - 1;
        }
        while (low <= high) {
            mid = (low + high) >> 1;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else if (nums[mid] < target) {
                low = mid + 1;
            }
        }
        return -1;
    }

    long long maximumImportance(int n, vector<vector<int>> &roads) {
        vector<int> degree(n, 0);
        for (auto &i: roads) {
            ++degree[i[0]];
            ++degree[i[1]];
        }
        sort(degree.begin(), degree.end());
        long long res = 0;
        for (int i = 0; i < n; ++i) {
            res += static_cast<long long>(degree[i]) * (i + 1);
        }
        return res;
    }

    int minimizeMax(vector<int> &nums, int p) {
        if (p == 0) {
            return 0;
        }
        sort(nums.begin(), nums.end());
        int low = 0;
        int high = nums.back() - nums.front();
        int mid;
        int res;
        while (low <= high) {
            mid = (low + high) >> 1;
            int count = 0;
            for (int i = 0; i < nums.size() - 1 && count < p;) {
                if (nums[i + 1] - nums[i] <= mid) {
                    count++;
                    i += 2;
                } else {
                    i++;
                }
            }
            if (count >= p) {
                res = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return res;
    }

};

int main() {
    return 0;
}