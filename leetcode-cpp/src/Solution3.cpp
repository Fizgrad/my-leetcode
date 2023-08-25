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
        vector<vector<vector<TreeNode *>>> dp(n + 1,
                                              vector<vector<TreeNode * >>(n + 1, vector<TreeNode *>()));
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

    bool search2(vector<int> &nums, int target) {
        int n = nums.size();
        int index = 1;
        for (int i = 1; i < n; ++i) {
            if (nums[i] == nums[i - 1]) {
                continue;
            } else {
                nums[index++] = nums[i];
            }
        }
        int low = 0;
        int high = index - 1;
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
            return true;
        } else if (target > nums[0]) {
            low = 0;
            high = cut;
        } else {
            low = cut + 1;
            high = index - 1;
        }
        while (low <= high) {
            mid = (low + high) >> 1;
            if (nums[mid] == target) {
                return true;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else if (nums[mid] < target) {
                low = mid + 1;
            }
        }
        return false;
    }

    int change(int amount, vector<int> &coins) {
        vector<vector<int>> cache(amount + 1, vector<int>(coins.size(), -1));
        auto f = [&](auto &&f, int money, int index) -> int {
            if (index >= coins.size() && money > 0) {
                return 0;
            }
            if (money < 0) {
                return 0;
            }
            if (money == 0) {
                return 1;
            }
            if (cache[money][index] != -1) {
                return cache[money][index];
            }
            int res = f(f, money, index + 1);
            int remains = money - coins[index];
            while (remains >= 0) {
                res += f(f, remains, index + 1);
                remains -= coins[index];
            }
            return cache[money][index] = res;
        };
        return f(f, amount, 0);
    }

    int uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid) {
        int n = obstacleGrid.size();
        int m = obstacleGrid.begin()->size();
        vector<int> res(m, 0);
        vector<int> temp(m, 0);
        for (int i = 0; i < m; ++i) {
            if (obstacleGrid[0][i] == 0)
                res[i] = 1;
            else break;
        }
        while (--n) {
            res.swap(temp);
            if (obstacleGrid[obstacleGrid.size() - n][0] == 0) {
                res[0] = temp[0];
            } else res[0] = 0;
            for (int i = 1; i < m; ++i) {
                if (obstacleGrid[obstacleGrid.size() - n][i] == 0)
                    res[i] = temp[i] + res[i - 1];
                else res[i] = 0;
            }
        }
        return res.back();
    }

    bool validPartition(vector<int> &nums) {
        int n = nums.size();
        vector<int> dp(n + 1, -1);
        auto f = [&](auto &&f, int i) -> bool {
            int n = nums.size();
            if (i == nums.size())
                return true;
            if (dp[i] != -1)
                return dp[i];

            if (i + 1 < n && nums[i] == nums[i + 1]) {
                if (f(f, i + 2))
                    return dp[i] = true;
            }

            if (i + 2 < n && nums[i] == nums[i + 2] and nums[i] == nums[i + 1]) {
                if (f(f, i + 3))
                    return dp[i] = true;
            }
            if (i + 2 < n && nums[i] + 1 == nums[i + 1] && nums[i] + 2 == nums[i + 2]) {
                if (f(f, i + 3))
                    return dp[i] = true;
            }
            return dp[i] = false;
        };
        return f(f, 0);
    }

    int findKthLargest(vector<int> &nums, int k) {
        std::sort(nums.begin(), nums.end());
        return *(nums.end() - k);
    }

    vector<vector<int>> updateMatrix(vector<vector<int>> &mat) {
        int n = mat.size();
        int m = mat.begin()->size();

        vector<vector<int>> res(n, vector<int>(m, -1));
        using coordinate = pair<int, int>;
        vector<coordinate> next;
        vector<coordinate> temp;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (mat[i][j] == 0) {
                    next.emplace_back(i, j);
                }
            }
        }
        int len = 0;

        constexpr int dx[4] = {0, 0, 1, -1};
        constexpr int dy[4] = {1, -1, 0, 0};

        while (!next.empty()) {
            for (auto [i, j]: next) {
                if (res[i][j] == -1) {
                    res[i][j] = len;
                    for (int k = 0; k < 4; ++k) {
                        int ii = i + dx[k];
                        int jj = j + dy[k];
                        if (ii >= 0 && jj >= 0 && ii < n && jj < m) {
                            if (res[ii][jj] == -1) {
                                temp.emplace_back(ii, jj);
                            }
                        }
                    }
                }
            }
            next.clear();
            next.swap(temp);
            ++len;
        }
        return res;
    }

    bool repeatedSubstringPattern(string s) {
        int n = s.length();
        for (int i = 1; i <= n / 2; ++i) {
            if (n % i == 0) {
                string substring = s.substr(0, i);
                for (int j = 1; j < n / i; ++j) {
                    if (s.substr(j * i, i) != substring) {
                        break;
                    } else {
                        if (j * i + i == n)
                            return true;
                    }
                }
            }
        }
        return false;
    }

    string reorganizeString(const string &s) {
        int len = s.size();
        vector<pair<int, char>> nums('z' - 'a' + 1);
        for (int i = 0; i < 26; ++i) {
            nums[i] = pair<int, char>(0, 'a' + i);
        }
        for (auto i: s) {
            ++nums[i - 'a'].first;
        }
        std::sort(nums.begin(), nums.end());
        if (nums.back().first * 2 > s.size() + 1) {
            return "";
        }
        string res(len, '1');
        int index = nums.size() - 1;
        for (int i = 0; i < len; i += 2) {
            res[i] = nums[index].second;
            --nums[index].first;
            if (nums[index].first == 0) {
                --index;
            }
        }
        for (int i = 1; i < len; i += 2) {
            res[i] = nums[index].second;
            --nums[index].first;
            if (nums[index].first == 0) {
                --index;
            }
        }
        return res;
    }

    string convertToTitle(int columnNumber) {
        using ll = long long int;
        auto pow = [](auto &&pow, ll x, ll n) -> ll {
            if (n == 0) {
                return 1;
            }
            if (n == 1) {
                return x;
            }
            ll temp = pow(pow, x, n / 2);
            ll res = temp * temp;
            if (n & 1) {
                res *= x;
            }
            return res;
        };
        float x = columnNumber - 1;
        int n = log(x * (25.0) / (26.0) + 1) / log(26);
        ll pre = 26 * (pow(pow, 26, n) - 1) / 25;
        ll remain = columnNumber - pre - 1;
        vector<char> res;
        while (remain >= 1) {
            res.push_back('A' + remain % 26);
            remain /= 26;
        }
        while (res.size() < n + 1) {
            res.push_back('A');
        }
        return {res.rbegin(), res.rend()};
    }

    bool isInterleave(const string &s1, const string &s2, const string &s3) {
        int n = s1.size();
        int m = s2.size();
        int res_len = s3.size();
        if (n + m != res_len) {
            return false;
        }
        vector<vector<short>> dp(n, vector<short>(m, -1));
        auto f = [&](auto &&f, int index1, int index2) -> int {
            if (index1 == n && index2 == m) {
                return 1;
            } else if (index2 == m) {
                return s3[index1 + index2] == s1[index1] && f(f, index1 + 1, index2);
            } else if (index1 == n) {
                return s3[index1 + index2] == s2[index2] && f(f, index1, index2 + 1);
            } else {
                if (dp[index1][index2] != -1) {
                    return dp[index1][index2];
                } else {
                    int res = false;
                    if (s3[index1 + index2] == s1[index1]) {
                        res |= f(f, index1 + 1, index2);
                    }
                    if (s3[index1 + index2] == s2[index2]) {
                        res |= f(f, index1, index2 + 1);
                    }
                    return dp[index1][index2] = res;
                }
            }

        };
        return f(f, 0, 0);
    }

};

int main() {
    return 0;
}