//
// Created by David Chen on 8/2/23.
//
#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <ranges>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

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
        vector<vector<int>> dp(n, vector<int>(n, -1));
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
                                              vector<vector<TreeNode *>>(n + 1, vector<TreeNode *>()));
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
            else
                break;
        }
        while (--n) {
            res.swap(temp);
            if (obstacleGrid[obstacleGrid.size() - n][0] == 0) {
                res[0] = temp[0];
            } else
                res[0] = 0;
            for (int i = 1; i < m; ++i) {
                if (obstacleGrid[obstacleGrid.size() - n][i] == 0)
                    res[i] = temp[i] + res[i - 1];
                else
                    res[i] = 0;
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

    int findLongestChain(vector<vector<int>> &pairs) {
        sort(pairs.begin(), pairs.end());
        int n = pairs.size();
        vector<int> dp(n, -1);
        auto f = [&](auto &&f, int index) {
            if (index >= n) {
                return 0;
            }
            if (index == n - 1) {
                return 1;
            }
            if (dp[index] != -1) {
                return dp[index];
            }
            int res = 1;
            for (int i = index + 1; i < n; ++i) {
                if (pairs[i][0] > pairs[index][1]) {
                    res = max(res, 1 + f(f, i));
                }
            }
            return dp[index] = res;
        };
        int res = 1;
        for (int i = 0; i < n; ++i) {
            res = max(res, f(f, i));
        }
        return res;
    }

    vector<int> countBits(int n) {
        vector<int> res(n + 1);
        for (int i = 0; i < n + 1; ++i) {
            res[i] = __builtin_popcount(i);
        }
        return res;
    }

    int minTaps(int n, vector<int> &ranges) {
        vector<int> a(n + 1, -1);
        for (int i = 0; i < ranges.size(); ++i) {
            int low = i - ranges[i];
            int high = i + ranges[i];
            if (low < 0) {
                a[0] = max(a[0], high);
            } else {
                a[low] = max(a[low], high);
            }
        }
        int res = 0;
        int reach = 0;
        int max_range = 0;
        for (int i = 0; i < n + 1; ++i) {
            if (i >= reach) {
                if (i > max_range) {
                    return -1;
                } else if (i < n) {
                    ++res;
                    reach = max(a[i], max_range);
                }
            }
            max_range = max(max_range, a[i]);
        }
        return res;
    }

    int uniquePaths(int m, int n) {
        long long ans = 1;
        for (int i = 1; i <= m - 1; ++i) {
            ans = ans * (n - 1 + i) / i;
        }
        return (int) ans;
    }

    ListNode *reverseBetween(ListNode *head, int left, int right) {
        if (left == right) {
            return head;
        }
        auto virtual_node = new ListNode(0);
        virtual_node->next = head;
        auto at = [&](int index) -> ListNode * {
            auto res = virtual_node;
            while (index--) {
                res = res->next;
            }
            return res;
        };
        auto left_left = at(left - 1);
        auto right_right = at(right + 1);
        auto left_pt = at(left);
        auto right_pt = at(right);


        auto next_pt = left_pt->next;
        auto next_next_pt = left_pt->next->next;
        auto pt = left_pt;
        while (pt != right_pt) {
            next_pt->next = pt;
            pt = next_pt;
            next_pt = next_next_pt;
            if (next_next_pt)
                next_next_pt = next_next_pt->next;
        }

        left_left->next = right_pt;
        left_pt->next = right_right;

        return virtual_node->next;
    }

    bool hasCycle(ListNode *head) {
        if (!head) {
            return false;
        }
        auto fast = head;
        auto slow = head;
        do {
            slow = slow->next;
            if (fast)
                fast = fast->next;
            else
                return false;
            if (fast)
                fast = fast->next;
            else
                return false;

        } while (fast != slow);

        //        int res = 0;
        //        fast = head;
        //        while (fast != slow) {
        //            slow = slow->next;
        //            fast = fast->next;
        //            ++res;
        //        }
        return true;
    }

    int bestClosingTime(const string &customers) {
        int n = customers.size();
        vector<int> prefix(n, 0);
        prefix[0] = (customers[0] == 'Y');
        for (int i = 1; i < n; ++i) {
            prefix[i] = prefix[i - 1] + (customers[i] == 'Y');
        }
        int closed = 0;
        int min_cost = prefix.back();
        for (int i = 0; i < n; ++i) {
            int cost = (i + 1) - prefix[i] + prefix.back() - prefix[i];
            if (cost < min_cost) {
                closed = i + 1;
                min_cost = cost;
            }
        }
        return closed;
    }

    vector<vector<int>> groupThePeople(vector<int> &groupSizes) {
        int n = groupSizes.size();
        vector<vector<int>> res;
        unordered_map<int, vector<int>> hm;
        for (int i = 0; i < n; ++i) {
            hm[groupSizes[i]].push_back(i);
            if (hm[groupSizes[i]].size() == groupSizes[i]) {
                res.push_back(hm[groupSizes[i]]);
                hm[groupSizes[i]] = vector<int>();
            }
        }
        return res;
    }

    int minDeletions(const string &s) {
        vector<int> freq('z' - 'a' + 1, 0);
        for (auto c: s) {
            ++freq[c - 'a'];
        }
        std::sort(freq.begin(), freq.end(), greater<>());
        int res = 0;
        int prev = 100000000;
        for (auto i: freq) {
            if (i == 0) {
                return res;
            }
            if (prev == 0) {
                res += i;
            } else {
                if (prev <= i) {
                    res += i - (prev - 1);
                    prev = prev - 1;
                } else {
                    prev = i;
                }
            }
        }
        return res;
    }

    int candy(vector<int> &ratings) {
        int n = ratings.size();

        if (n == 1) {
            return ratings.size();
        }
        vector<int> res(n, 0);
        auto dfs = [&](auto &&dfs, int count, int index) {
            int dx[2] = {1, -1};
            if (count < res[index]) {
                return;
            } else {
                res[index] = count;
            }
            for (int i = 0; i < 2; ++i) {
                int x = dx[i] + index;
                if (x >= 0 && x < n && ratings[index] < ratings[x]) {
                    dfs(dfs, count + 1, x);
                }
            }
        };
        if (ratings[0] <= ratings[1]) {
            dfs(dfs, 1, 0);
        }
        if (ratings[n - 1] <= ratings[n - 2]) {
            dfs(dfs, 1, n - 1);
        }
        for (int i = 1; i < n - 1; ++i) {
            if (ratings[i] <= ratings[i - 1] && ratings[i] <= ratings[i + 1]) {
                res[i] = 1;
                dfs(dfs, 1, i);
            }
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += res[i];
        }
        return ans;
    }

    int minCostConnectPoints(vector<vector<int>> &points) {
        int n = points.size();
        auto cal_dis = [&](int i, int j) {
            return ::abs(points[i][0] - points[j][0]) + ::abs(points[i][1] - points[j][1]);
        };
        vector<int> parents(n + 1, 0);
        vector<int> sizes(parents.size(), 1);
        for (int i = 0; i < parents.size(); ++i) {
            parents[i] = i;
        }

        auto uf_find = [&](int i) {
            int next = parents[i];
            while (next != i) {
                i = next;
                next = parents[i];
            }
            return i;
        };

        auto uf_union = [&](int a, int b) {
            int pa = uf_find(a);
            int pb = uf_find(b);
            if (sizes[pa] > sizes[pb]) {
                parents[pb] = pa;
                sizes[pa] += sizes[pb];
            } else {
                parents[pa] = pb;
                sizes[pb] += sizes[pa];
            }
        };

        using dis_a_b = pair<int, pair<int, int>>;
        vector<dis_a_b> data;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int dis = cal_dis(i, j);
                data.emplace_back(dis_a_b(dis, pair<int, int>(i, j)));
            }
        }

        sort(begin(data), end(data), [](const dis_a_b &a, const dis_a_b &b) {
            return a.first < b.first;
        });

        int res = 0;
        for (int i = 0; i < data.size(); ++i) {
            dis_a_b tmp = data[i];
            int a = tmp.second.first;
            int b = tmp.second.second;

            if (uf_find(a) == uf_find(b)) {
                continue;
            } else {
                uf_union(a, b);
                res += tmp.first;
            }
        }
        return res;
    }

    int minimumEffortPath(vector<vector<int>> &heights) {
        int rows = heights.size(), cols = heights[0].size();
        vector<vector<int>> dist(rows, vector<int>(cols, INT_MAX));
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>>
                minHeap;
        minHeap.emplace(0, 0, 0);
        dist[0][0] = 0;

        int directions[4][2] = {{0, 1},
                                {0, -1},
                                {1, 0},
                                {-1, 0}};

        while (!minHeap.empty()) {
            auto [effort, x, y] = minHeap.top();
            minHeap.pop();

            if (effort > dist[x][y]) continue;

            if (x == rows - 1 && y == cols - 1) return effort;

            for (auto &dir: directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                    int new_effort = max(effort, abs(heights[x][y] - heights[nx][ny]));
                    if (new_effort < dist[nx][ny]) {
                        dist[nx][ny] = new_effort;
                        minHeap.emplace(new_effort, nx, ny);
                    }
                }
            }
        }
        return -1;
    }

    int findDuplicate(vector<int> &nums) {
        int n = nums.size();
        int fast = 0;
        int slow = 0;

        do {
            slow = nums[slow];
            fast = nums[fast];
            fast = nums[fast];
        } while (fast != slow);
        fast = 0;
        do {
            slow = nums[slow];
            fast = nums[fast];
        } while (fast != slow);
        return slow;
    }

    int numIdenticalPairs(vector<int> &nums) {
        vector<int> size(101, 0);
        int res = 0;
        for (auto i: nums) {
            res += size[i];
            ++size[i];
        }
        return res;
    }

    int findContentChildren(vector<int> &g, vector<int> &s) {
        std::sort(g.begin(), g.end());
        std::sort(s.begin(), s.end());
        int res = 0;
        auto j = g.rbegin();
        for (auto i = s.rbegin(); i != s.rend() && j != g.rend(); ++j) {
            if (*i >= *j) {
                ++res;
                ++i;
            }
        }
        return res;
    }

    vector<vector<int>> findMatrix(vector<int> &nums) {
        vector<int> times(201, 0);
        vector<vector<int>> res(0);
        int size = 0;
        for (auto i: nums) {
            if (size == times[i]) {
                res.emplace_back(0);
                ++size;
            }
            res[times[i]].push_back(i);
            ++times[i];
        }
        return res;
    }

    int numberOfBeams(vector<string> &bank) {
        auto Count = [](const std::string &line) -> int {
            return std::accumulate(line.begin(), line.end(), 0,
                                   [](int acc, char ch) -> int { return acc + (ch == '1' ? 1 : 0); });
        };

        int totalBeams = 0, previousCount = 0, isFirst = 1;
        std::for_each(bank.begin(), bank.end(),
                      [&](const std::string &row) {
                          if (!isFirst) {
                              int currentCount = Count(row);
                              if (currentCount) {
                                  totalBeams += previousCount * currentCount;
                                  previousCount = currentCount;
                              }
                          } else {
                              previousCount = Count(row);
                              isFirst = 0;
                          }
                      });

        return totalBeams;
    }

    int minOperations(vector<int> &nums) {
        unordered_map<int, int> times;
        int res = 0;
        std::for_each(nums.begin(), nums.end(), [&](int i) -> void { ++times[i]; });
        for (auto [_, i]: times) {
            if (i == 1) {
                return -1;
            }
            res += (i + 2) / 3;
        }
        return res;
    }

    int lengthOfLIS(vector<int> &nums) {
        vector<int> subsequence;
        auto f = [&](int key) -> int {
            int i = 0;
            int j = subsequence.size();
            int ans = j;
            while (j >= i) {
                int mid = (i + j) >> 1;
                if (subsequence[mid] > key) {
                    ans = mid;
                    j = mid - 1;
                } else if (subsequence[mid] == key) {
                    return mid;
                } else {
                    i = mid + 1;
                }
            }
            return ans;
        };
        for (auto i: nums) {
            if (subsequence.empty() || i > subsequence.back()) {
                subsequence.push_back(i);
                continue;
            }
            auto index = f(i);
            subsequence[index] = i;
        }
        return subsequence.size();
    }

    int jobScheduling(vector<int> &startTime, vector<int> &endTime, vector<int> &profit) {
        int n = startTime.size();
        vector<tuple<int, int, int>> inputs(n);
        for (int i = 0; i < n; ++i) {
            inputs[i] = std::make_tuple(startTime[i], endTime[i], profit[i]);
        }
        std::sort(inputs.begin(), inputs.end());
        vector<int> dp(n + 1, 0);
        auto f = [&](int end_time) -> int {
            int i = 0;
            int j = n - 1;
            int ans = -1;
            while (i <= j) {
                int mid = (i + j) >> 1;
                if (get<0>(inputs[mid]) > end_time) {
                    j = mid - 1;
                    ans = mid;
                } else if (get<0>(inputs[mid]) == end_time) {
                    j = mid - 1;
                    ans = mid;
                } else {
                    i = mid + 1;
                }
            }
            return ans;
        };
        for (int i = n - 1; i >= 0; --i) {
            int index = f(get<1>(inputs[i]));
            if (index == -1) {
                dp[i] = max(dp[i + 1], get<2>(inputs[i]));
            } else {
                dp[i] = max(dp[i + 1], get<2>(inputs[i]) + dp[index]);
            }
        }
        return dp[0];
    }

    int numberOfArithmeticSlices(vector<int> &nums) {
        int n = nums.size();
        int res = 0;
        vector<unordered_map<int64_t, int>> dp(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int64_t dif = nums[i] - nums[j];
                res += dp[j][dif];
                dp[i][dif] += dp[j][dif] + 1;
            }
        }
        return res;
    }

    int rangeSumBST(TreeNode *root, int low, int high) {
        auto sumSubTree = [&](auto &&sumSubTree, TreeNode *node, int low, int high) -> int {
            int res = 0;
            if (node == nullptr) {
                return res;
            }
            int val = node->val;
            if (val >= low && val <= high) {
                res += val;
                res += sumSubTree(sumSubTree, node->left, low, high);
                res += sumSubTree(sumSubTree, node->right, low, high);
            } else if (val > high) {
                res += sumSubTree(sumSubTree, node->left, low, high);
            } else if (val < low) {
                res += sumSubTree(sumSubTree, node->right, low, high);
            }
            return res;
        };
        return sumSubTree(sumSubTree, root, low, high);
    }

    bool leafSimilar(TreeNode *root1, TreeNode *root2) {
        auto f = [](auto &&f, TreeNode *root, vector<int> &num) {
            if (root == nullptr) {
                return;
            }
            if (root->left == nullptr && root->right == nullptr) {
                num.push_back(root->val);
            }
            f(f, root->left, num);
            f(f, root->right, num);
        };
        vector<int> num1;
        vector<int> num2;
        f(f, root1, num1);
        f(f, root2, num2);
        return num1.size() == num2.size() && std::equal(num1.begin(), num1.end(), num2.begin(), num2.end());
    }

    int amountOfTime(TreeNode *root, int start) {
        int ans = 0;
        auto f = [&](auto &&f, TreeNode *node, bool encounterStart, int len) -> pair<int, bool> {
            if (node == nullptr) {
                return {0, 0};
            }
            if (node->val == start) {
                f(f, node->left, true, 1);
                f(f, node->right, true, 1);
                return {1, 1};
            }
            if (encounterStart) {
                ans = max(ans, len);
                f(f, node->left, true, len + 1);
                f(f, node->right, true, len + 1);
                return {0, 1};
            } else {
                pair<int, bool> left = f(f, node->left, encounterStart, len + 1);
                pair<int, bool> right = f(f, node->right, encounterStart, len + 1);
                if (left.second) {
                    ans = max(ans, left.first + right.first);
                    return {left.first + 1, 1};
                } else if (right.second) {
                    ans = max(ans, left.first + right.first);
                    return {right.first + 1, 1};
                } else {
                    return {max(left.first, right.first) + 1, 0};
                }
            }
        };
        f(f, root, false, 0);
        return ans;
    }

    int maxAncestorDiff(TreeNode *root) {
        int ans = 0;
        auto dfs = [&](auto &&dfs, TreeNode *node, int max_num, int min_num) {
            if (!node) return;
            max_num = max(max_num, node->val);
            min_num = min(min_num, node->val);
            ans = max(ans, std::abs(max_num - min_num));
            dfs(dfs, node->left, max_num, min_num);
            dfs(dfs, node->right, max_num, min_num);
        };
        dfs(dfs, root, root->val, root->val);
        return ans;
    }

    bool halvesAreAlike(const string &s) {
        int sum = 0;
        auto i = 0;
        for (; i < s.size() >> 1; ++i) {
            if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' || s[i] == 'E' ||
                s[i] == 'I' || s[i] == 'O' || s[i] == 'U')
                ++sum;
        }
        for (; i < s.size(); ++i) {
            if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' || s[i] == 'E' ||
                s[i] == 'I' || s[i] == 'O' || s[i] == 'U')
                --sum;
        }
        return !sum;
    }

    int minSteps(const string &s, const string &t) {
        int res = 0;
        int nums[26] = {0};
        for (auto i: s) {
            ++nums[i - 'a'];
        }
        for (auto i: t) {
            --nums[i - 'a'];
        }
        for (auto i: nums) {
            res += (i > 0 ? i : 0);
        }
        return res;
    }

    bool closeStrings(const string &word1, const string &word2) {
        if (word2.size() != word1.size()) {
            return false;
        }
        int nums1[26] = {0};
        int nums2[26] = {0};
        for (auto i: word1) {
            ++nums1[i - 'a'];
        }
        for (auto i: word2) {
            ++nums2[i - 'a'];
        }
        for (int i = 0; i < 26; ++i) {
            if ((nums1[i] > 0 && nums2[i] == 0) || (nums2[i] > 0 && nums1[i] == 0)) {
                return false;
            }
        }
        unordered_map<int, int> num_times;
        for (auto i: nums1) {
            ++num_times[i];
        }
        for (auto i: nums2) {
            --num_times[i];
        }
        return std::all_of(num_times.begin(), num_times.end(), [](auto i) {
            return i.second == 0;
        });
    }

    vector<vector<int>> findWinners(vector<vector<int>> &matches) {
        vector<vector<int>> answer(2);
        unordered_set<int> all;
        unordered_map<int, int> lost;
        for (auto &i: matches) {
            all.insert(i[0]);
            all.insert(i[1]);
            ++lost[i[1]];
        }
        for (auto i: all) {
            if (lost.find(i) == lost.end()) {
                answer[0].push_back(i);
            }
        }
        for (auto [i, t]: lost) {
            if (t == 1) {
                answer[1].push_back(i);
            }
        }
        std::sort(answer[0].begin(), answer[0].end());
        std::sort(answer[1].begin(), answer[1].end());
        return answer;
    }

    bool uniqueOccurrences(vector<int> &arr) {
        unordered_map<int, int> map;
        for (auto i: arr) {
            map[i] += 1;
        }
        unordered_set<int> set;
        for (auto [i, time]: map) {
            if (set.find(time) == set.end()) {
                set.insert(time);
            } else {
                return false;
            }
        }
        return true;
    }

    int climbStairs(int n) {
        unordered_map<int, int> res;
        auto f = [&](auto &&f, int n) -> int {
            if (n <= 0) {
                return 0;
            }
            if (n == 1) {
                return 1;
            }
            if (n == 2) {
                return 2;
            }
            if (res.find(n) != res.end()) {
                return res[n];
            }
            return res[n] = f(f, n - 1) + f(f, n - 2);
        };
        return f(f, n);
    }

    int minFallingPathSum(vector<vector<int>> &matrix) {
        int n = matrix.size();
        vector<vector<int>> dp(2, vector<int>(n, 0));
        for (int i = 0; i < n; ++i) {
            vector<int> &res = dp[i % 2];
            vector<int> &prev = dp[1 - i % 2];
            for (int j = 0; j < n; ++j) {
                res[j] = prev[j] + matrix[i][j];
                if (j > 0) res[j] = min(res[j], prev[j - 1] + matrix[i][j]);
                if (j + 1 < n) res[j] = min(res[j], prev[j + 1] + matrix[i][j]);
            }
        }
        return *std::min_element(dp[(n - 1) % 2].begin(), dp[(n - 1) % 2].end());
    }

    int sumSubarrayMins(vector<int> &arr) {
        constexpr int Mod = 1000000000 + 7;
        int res = 0;
        auto updateRes = [&](long long num) {
            res = (res + num % Mod) % Mod;
        };
        int n = arr.size();
        stack<pair<int, int>> stack;
        for (int i = 0; i < n; ++i) {
            if (stack.empty()) {
                stack.emplace(arr[i], i);
            } else {
                while (stack.size() && arr[stack.top().second] >= arr[i]) {
                    auto [num, index] = stack.top();
                    stack.pop();
                    updateRes((i - index) * num);
                }
                if (stack.empty()) {
                    stack.emplace(arr[i] * static_cast<long long>(i + 1) % Mod, i);
                } else {
                    stack.emplace(arr[i] * static_cast<long long>(i - stack.top().second) % Mod, i);
                }
            }
        }
        while (stack.size()) {
            auto [num, index] = stack.top();
            stack.pop();
            updateRes(static_cast<long long>(n - index) % Mod * num);
        }
        return res;
    }

    int rob(vector<int> &nums) {
        auto n = nums.size();
        vector<int> dp(3, 0);
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return nums[0];
        }
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (auto i = 2; i < n; ++i) {
            dp[i % 3] = max(dp[(i - 1) % 3], dp[(i - 2) % 3] + nums[i]);
        }
        return dp[(n - 1) % 3];
    }

    vector<int> findErrorNums(vector<int> &nums) {
        auto n = nums.size();
        int xor_sum = 0;
        vector<int> times(n + 1, 0);
        int rep = 0;
        for (auto i: nums) {
            xor_sum ^= i;
            if (!rep && times[i] == 1) {
                rep = i;
            }
            ++times[i];
        }
        for (int i = 0; i <= n; ++i) {
            xor_sum ^= i;
        }
        return {rep, xor_sum ^ rep};
    }

    int maxLength(vector<string> &arr) {
        for (auto &i: arr) {
            vector<int> nums(26, 0);
            for (auto j: i) {
                if (nums[j - 'a']) {
                    i = "";
                    break;
                }
                nums[j - 'a'] += 1;
            }
        }
        int res = 0;
        vector<int> nums(26, 0);
        auto n = arr.size();
        auto f = [&](auto &&f, int index) -> void {
            res = max(res, std::accumulate(nums.begin(), nums.end(), 0));
            if (index >= n) return;
            f(f, index + 1);
            for (auto i: arr[index]) {
                if (nums[i - 'a']) {
                    return;
                }
            }
            for (auto i: arr[index]) {
                nums[i - 'a'] = 1;
            }
            f(f, index + 1);
            for (auto i: arr[index]) {
                nums[i - 'a'] = 0;
            }
        };
        f(f, 0);
        return res;
    }

    int pseudoPalindromicPaths(TreeNode *root) {
        vector<int> nums(10, 0);
        int res = 0;
        auto f = [&](auto &&f, TreeNode *node) {
            if (node == nullptr) return;
            nums[node->val] = 1 - nums[node->val];
            if (node->left) f(f, node->left);
            if (node->right) f(f, node->right);
            if (!node->left && !node->right && std::accumulate(nums.begin(), nums.end(), 0) <= 1) ++res;
            nums[node->val] = 1 - nums[node->val];
        };
        f(f, root);
        return res;
    }

    int longestCommonSubsequence(const string &text1, const string &text2) {
        auto n = text1.size();
        auto m = text2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                if (text2[j - 1] == text1[i - 1]) {
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1);
                }
            }
        }
        return dp[n][m];
    }

    int findPaths(int m, int n, int N, int x, int y) {
        const int M = 1000000000 + 7;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp[x][y] = 1;
        int count = 0;

        for (int moves = 1; moves <= N; moves++) {
            vector<vector<int>> temp(m, vector<int>(n, 0));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == m - 1) count = (count + dp[i][j]) % M;
                    if (j == n - 1) count = (count + dp[i][j]) % M;
                    if (i == 0) count = (count + dp[i][j]) % M;
                    if (j == 0) count = (count + dp[i][j]) % M;
                    temp[i][j] = (((i > 0 ? dp[i - 1][j] : 0) + (i < m - 1 ? dp[i + 1][j] : 0)) % M +
                                  ((j > 0 ? dp[i][j - 1] : 0) + (j < n - 1 ? dp[i][j + 1] : 0)) % M) %
                                 M;
                }
            }
            dp = temp;
        }
        return count;
    }

    int kInversePairs(int n, int k) {
        const int MOD = 1000000007;
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(k + 1, 0));
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j <= k; ++j) {
                if (j == 0) {
                    dp[i][j] = 1;
                } else {
                    int val = (dp[i - 1][j] + MOD - (j - i >= 0 ? dp[i - 1][j - i] : 0)) % MOD;
                    dp[i][j] = (dp[i][j - 1] + val) % MOD;
                }
            }
        }
        return (dp[n][k] + MOD - (k > 0 ? dp[n][k - 1] : 0)) % MOD;
    }

    int numSubmatrixSumTarget(vector<vector<int>> &matrix, int target) {
        int ans = 0;
        int n = matrix.size(), m = matrix[0].size();
        for (int left = 0; left < m; left++) {
            vector<int> pre(n, 0);
            for (int right = left; right < m; right++) {
                for (int i = 0; i < n; i++) {
                    pre[i] += matrix[i][right];
                }

                for (int i = 0; i < n; i++) {
                    int sum = 0;
                    for (int j = i; j < n; j++) {
                        sum += pre[j];
                        if (sum == target) {
                            ans += 1;
                        }
                    }
                }
            }
        }
        return ans;
    }

    int evalRPN(vector<string> &tokens) {
        vector<int> stack;
        stack.push_back(0);
        for (auto &i: tokens) {
            if (i.size() == 1 && (i[0] < '0' || i[0] > '9')) {
                int a = stack.back();
                stack.pop_back();
                int b = stack.back();
                stack.pop_back();
                switch (i[0]) {
                    case '+':
                        stack.push_back(a + b);
                        break;
                    case '-':
                        stack.push_back(b - a);
                        break;
                    case '*':
                        stack.push_back(a * b);
                        break;
                    case '/':
                        stack.push_back(b / a);
                        break;
                }
            } else if (std::all_of(i.begin(), i.end(), [&](const auto &item) {
                           return (item >= '0' && item <= '9') || item == '-';
                       })) {
                stack.push_back(stoi(i));
                continue;
            }
        }
        return stack.back();
    }

    vector<int> dailyTemperatures(vector<int> &temperatures) {
        auto n = temperatures.size();
        vector<int> ans(n, 0);
        stack<pair<int, int>> s;
        for (auto i = 0; i < n; ++i) {
            if (s.empty() || temperatures[i] <= s.top().first) {
                s.emplace(temperatures[i], i);
            } else {
                while (!s.empty() && temperatures[i] > s.top().first) {
                    auto [top_t, top_i] = s.top();
                    s.pop();
                    ans[top_i] = i - top_i;
                }
                s.emplace(temperatures[i], i);
            }
        }
        return ans;
    }

    vector<vector<int>> divideArray(vector<int> &nums, int k) {
        std::sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        for (auto i = 0; i < nums.size(); i += 3) {
            if (nums[i + 2] - nums[i] <= k) {
                ans.push_back({nums[i], nums[i + 1], nums[i + 2]});
            } else {
                return {};
            }
        }
        return ans;
    }

    vector<int> sequentialDigits(int low, int high) {
        vector<int> ans;
        for (int len = 2; len <= 9; ++len) {
            for (int k = 1; k <= 10 - len; ++k) {
                int num = 0;
                int j = k;
                for (int i = 0; i < len; ++i) {
                    num *= 10;
                    num += j;
                    ++j;
                }
                if (num >= low) {
                    if (num <= high) {
                        ans.push_back(num);
                    } else {
                        return ans;
                    }
                }
            }
        }
        return ans;
    }

    int maxSumAfterPartitioning(vector<int> &arr, int k) {
        int n = arr.size();
        vector<int> dp(n, -1);
        auto f = [&](auto &&f, int last_index) {
            if (dp[last_index] != -1)
                return dp[last_index];
            if (last_index < k) {
                return dp[last_index] = (last_index + 1) * (*max_element(arr.begin(), arr.begin() + last_index + 1));
            }
            int res = 0;
            for (int i = 1; i <= k; ++i) {
                res = max(f(f, last_index - i) +
                                  (i) * (*max_element(arr.begin() + last_index - i + 1, arr.begin() + last_index + 1)),
                          res);
            }
            return dp[last_index] = res;
        };
        return f(f, n - 1);
    }

    string minWindow(const string &s, const string &t) {
        auto i = s.begin();
        auto j = i;
        vector<int> nums('z' - 'A' + 1, 0);
        for (auto c: t) {
            ++nums[c - 'A'];
        }
        int res = std::numeric_limits<std::int32_t>::max() >> 1;
        auto res_start = s.begin();
        auto res_end = s.begin();
        while (j <= s.end()) {
            bool flag = false;
            if (std::all_of(nums.begin(), nums.end(), [&](const auto &item) {
                    return item <= 0;
                })) {
                flag = true;
                if (res > static_cast<int>(j - i)) {
                    res_start = i;
                    res_end = j;
                    res = static_cast<int>(j - i);
                }
            }
            if (i < j && j == s.end()) {
                ++nums[*i - 'A'];
                ++i;
                continue;
            }
            if (j == i) {
                if (j == s.end()) break;
                --nums[*j - 'A'];
                ++j;
                continue;
            }
            if (flag) {
                ++nums[*i - 'A'];
                ++i;
                continue;
            } else {
                --nums[*j - 'A'];
                ++j;
            }
        }
        return {res_start, res_end};
    }

    int firstUniqChar(const std::string &s) {
        int freq[26] = {};
        for (char c: s) ++freq[c - 'a'];
        for (int i = 0; i < s.size(); ++i)
            if (freq[s[i] - 'a'] == 1) return i;
        return -1;
    }

    int maxResult(vector<int> &nums, int k) {
        auto n = nums.size();
        if (n == 1) {
            return nums[0];
        }
        priority_queue<pair<int, int>> heap;
        heap.emplace(nums[n - 1], n - 1);
        for (int i = n - 2; i >= 0; --i) {
            auto [dp, index] = heap.top();
            while (index > i + k) {
                heap.pop();
                index = heap.top().second;
                dp = heap.top().first;
            }
            heap.emplace(dp + nums[i], i);
            if (i == 0) {
                return dp + nums[i];
            }
        }
        return -1;
    }

    string maximumTime(string time) {
        if (time[0] == '?') time[0] = (time[1] <= '3' || time[1] == '?') ? '2' : '1';
        if (time[1] == '?') time[1] = (time[0] == '2') ? '3' : '9';
        if (time[3] == '?') time[3] = '5';
        if (time[4] == '?') time[4] = '9';
        return time;
    }

    vector<vector<string>> groupAnagrams(vector<string> &strs) {
        using CharFreq = array<unsigned char, 26>;
        auto encode = [](const string &s) {
            CharFreq res{};
            for (auto c: s) ++res[c - 'a'];
            return res;
        };
        map<CharFreq, vector<string>> res;
        for (auto &i: strs) res[encode(i)].emplace_back(i);
        return std::accumulate(res.begin(), res.end(), vector<vector<string>>(),
                               [](vector<vector<string>> sum, auto item) {
                                   sum.emplace_back(item.second);
                                   return sum;
                               });
    }

    string frequencySort(const string &s) {
        string res(s);
        int nums[CHAR_MAX - CHAR_MIN + 1] = {};
        for (auto i: s) ++nums[i];
        std::sort(res.begin(), res.end(), [&](const auto a, const auto b) {
            return nums[a] > nums[b] || (nums[a] == nums[b] && a < b);
        });
        return std::move(res);
    }

    int numSquares(int n) {
        auto is_square = [](int n) {
            int sqrt_n = (int) (sqrt(n));
            return (sqrt_n * sqrt_n == n);
        };
        if (is_square(n)) return 1;
        while ((n & 3) == 0) n >>= 2;
        if ((n & 7) == 7) return 4;
        int sqrt_n = (int) (sqrt(n));
        for (int i = 1; i <= sqrt_n; i++) {
            if (is_square(n - i * i)) {
                return 2;
            }
        }
        return 3;
    }

    vector<int> largestDivisibleSubset(vector<int> &nums) {
        int n = nums.size(), res = 1, num = -1;
        vector<int> ans;
        sort(nums.begin(), nums.end());
        vector<int> dp(n, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (!(nums[i] % nums[j]) && dp[i] < dp[j] + 1) {
                    res = max(res, dp[i] = dp[j] + 1);
                }
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            if (res == dp[i] && (num == -1 || !(num % nums[i]))) {
                ans.push_back(num = nums[i]);
                res--;
            }
        }
        return ans;
    }

    int countSubstrings(string s) {
        auto palindromeCount = [&](const string &s, int left, int right) {
            int count = 0;
            while (left >= 0 && right < s.length() && s[left] == s[right]) {
                --left;
                ++right;
                ++count;
            }
            return count;
        };
        int n = s.length(), ans = 0;
        for (int i = 0; i < n; ++i) {
            int even = palindromeCount(s, i, i + 1);
            int odd = palindromeCount(s, i, i);
            ans += even + odd;
        }
        return ans;
    }

    int cherryPickup(vector<vector<int>> &grid) {
        int rows = grid.size(), cols = grid[0].size();
        vector<vector<int>> dp(cols, vector<int>(cols, 0));
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                dp[i][j] = i == j ? grid[rows - 1][i] : grid[rows - 1][i] + grid[rows - 1][j];
            }
        }
        for (int row = rows - 2; row >= 0; row--) {
            vector<vector<int>> newDp(cols, vector<int>(cols, 0));
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < cols; j++) {
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ni = i + di, nj = j + dj;
                            if (ni >= 0 && ni < cols && nj >= 0 && nj < cols) {
                                int cherries = i == j ? grid[row][i] : grid[row][i] + grid[row][j];
                                newDp[i][j] = max(newDp[i][j], dp[ni][nj] + cherries);
                            }
                        }
                    }
                }
            }
            dp = std::move(newDp);
        }
        return dp[0][cols - 1];
    }

    vector<int> majorityElement(vector<int> &nums) {
        // 229. Majority Element II
        auto n = nums.size();
        if (n == 1 || (n == 2 && nums[0] == nums[1])) return {nums[0]};
        if (n == 2) return {nums[0], nums[1]};
        int candidate1 = nums[0];
        int candidate2 = std::numeric_limits<std::int32_t>::min();
        int times1 = 2;
        int times2 = 0;
        for (int i = 1; i < (n / 3) * 3; ++i) {
            if (nums[i] == candidate1) {
                --times2;
                times1 += 2;
                if (times2 < 0) {
                    candidate2 = std::numeric_limits<std::int32_t>::min();
                    times2 = 0;
                }
                continue;
            } else if (nums[i] == candidate2) {
                --times1;
                times2 += 2;
                if (times1 < 0) {
                    candidate1 = std::numeric_limits<std::int32_t>::min();
                    times1 = 0;
                }
                continue;
            } else if (candidate1 == std::numeric_limits<std::int32_t>::min()) {
                candidate1 = nums[i];
                times1 = 2;
                --times2;
                if (times2 < 0) {
                    candidate2 = std::numeric_limits<std::int32_t>::min();
                    times2 = 0;
                }
                continue;
            } else if (candidate2 == std::numeric_limits<std::int32_t>::min()) {
                candidate2 = nums[i];
                times2 = 2;
                --times1;
                if (times1 < 0) {
                    candidate1 = std::numeric_limits<std::int32_t>::min();
                    times1 = 0;
                }
                continue;
            } else {
                --times1;
                --times2;
                if (times1 < 0) {
                    candidate1 = nums[i];
                    times1 = 2;
                    if (times2 < 0) {
                        candidate2 = std::numeric_limits<std::int32_t>::min();
                        times2 = 0;
                    }
                } else if (times2 < 0) {
                    candidate2 = nums[i];
                    times2 = 2;
                }
            }
        }
        vector<int> res;
        times1 = 0;
        for (auto i: nums) times1 += i == candidate1;
        times2 = 0;
        for (auto i: nums) times2 += i == candidate2;
        if (times1 * 3 > nums.size()) res.push_back(candidate1);
        if (times2 * 3 > nums.size()) res.push_back(candidate2);
        for (auto i = (n / 3) * 3; i < n; ++i) {
            if (std::find(res.begin(), res.end(), nums[i]) != res.end()) continue;
            times1 = 0;
            for (auto k: nums) times1 += (k == nums[i]);
            if (times1 * 3 > nums.size()) res.push_back(nums[i]);
        }
        return res;
    }

    string firstPalindrome(vector<string> &words) {
        auto isPalindrome = [](const string &i) {
            auto left = i.begin();
            auto right = i.end() - 1;
            while (right > left) {
                if (*left == *right) {
                    --right;
                    ++left;
                } else
                    return false;
            }
            return true;
        };
        for (auto &i: words) {
            if (isPalindrome(i)) return i;
        }
        return "";
    }

    int maxProduct(const string &s) {
        class StringIndexAdapter {
        public:
            StringIndexAdapter(const std::string &str, const std::vector<int> &indices)
                : str_(str), indices_(indices) {}

            class Iterator {
            public:
                Iterator(const StringIndexAdapter *adapter, size_t pos)
                    : adapter_(adapter), pos_(pos) {}

                Iterator &operator++() {
                    ++pos_;
                    return *this;
                }

                Iterator &operator--() {
                    --pos_;
                    return *this;
                }

                char operator*() const { return adapter_->str_[adapter_->indices_[pos_]]; }

                bool operator!=(const Iterator &other) const { return pos_ != other.pos_; }

                bool operator==(const Iterator &other) const { return pos_ == other.pos_; }

                bool operator<(const Iterator &other) const { return pos_ < other.pos_; }

                bool operator>(const Iterator &other) const { return pos_ > other.pos_; }

                bool operator<=(const Iterator &other) const { return pos_ <= other.pos_; }

                bool operator>=(const Iterator &other) const { return pos_ >= other.pos_; }

            private:
                const StringIndexAdapter *adapter_;
                size_t pos_;
            };

            Iterator begin() const { return {this, 0}; }

            Iterator end() const { return {this, indices_.size()}; }

            const std::string &str_;
            const std::vector<int> indices_;
        };

        auto isPalindrome = [](const auto &i) -> bool {
            if (i.begin() == i.end()) {
                return false;
            }
            auto left = i.begin();
            auto right = i.end();
            --right;
            while (right > left) {
                if (*left == *right) {
                    --right;
                    ++left;
                } else
                    return false;
            }
            return true;
        };
        vector<StringIndexAdapter> Palindromes;

        auto f = [&](auto &&f, vector<int> &indices_, int index) {
            if (isPalindrome(StringIndexAdapter(s, indices_))) Palindromes.emplace_back(s, indices_);
            if (index == s.size()) return;
            indices_.push_back(index);
            ++index;
            f(f, indices_, index);
            indices_.pop_back();
            f(f, indices_, index);
        };

        auto checkIsDisjoint = [](const StringIndexAdapter &a, const StringIndexAdapter &b) {
            for (auto i: a.indices_)
                for (auto j: b.indices_)
                    if (i == j) return false;
            return true;
        };

        vector<int> indices;
        f(f, indices, 0);

        int res = 0;
        for (auto i = Palindromes.begin(); i != Palindromes.end(); ++i) {
            for (auto j = i + 1; j != Palindromes.end(); ++j) {
                if (checkIsDisjoint(*i, *j)) {
                    res = max(res, static_cast<int>(i->indices_.size() * j->indices_.size()));
                }
            }
        }
        return res;
    }

    vector<int> rearrangeArray(vector<int> &nums) {
        int pos = 0;
        int n = nums.size();
        int pos_index = 0;
        int neg_index = 0;
        vector<int> res(n, 0);
        while (pos < n) {
            if (pos & 1) {
                while (nums[neg_index] >= 0) { ++neg_index; }
                res[pos++] = nums[neg_index++];
            } else {
                while (nums[pos_index] < 0) { ++pos_index; }
                res[pos++] = nums[pos_index++];
            }
        }
        return res;
    }

    int wiggleMaxLength(vector<int> &nums) {
        int n = static_cast<int>(std::unique(nums.begin(), nums.end()) - nums.begin());
        if (n == 1) return 1;
        if (n == 2) return 2;
        int prev_diff = 0;
        int res = 2;
        for (int i = 1; i < n; ++i) {
            int diff = nums[i] - nums[i - 1];
            if (diff * prev_diff >= 0) {
            } else {
                res += 1;
            }
            prev_diff = diff;
        }
        return res;
    }

    string addSpaces(const string &s, vector<int> &spaces) {
        string res;
        res.reserve(s.size() + spaces.size());
        int index = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (index < spaces.size() && i == spaces[index]) {
                res.push_back(' ');
                ++index;
            }
            res.push_back(s[i]);
        }
        return res;
    }

    vector<int> eventualSafeNodes(vector<vector<int>> &graph) {
        int n = graph.size();
        vector<int> res;
        vector<int> dp(n, -1);
        auto dfs = [&](auto &&dfs, int node) {
            if (dp[node] != -1) return dp[node];
            dp[node] = 0;
            for (auto i: graph[node]) {
                if (dfs(dfs, i) == 0) {
                    return dp[node] = 0;
                }
            }
            return dp[node] = 1;
        };
        for (int i = 0; i < n; ++i) {
            if (dfs(dfs, i)) res.push_back(i);
        }
        return res;
    }

    int maximumInvitations(vector<int> &favorites) {
        int n = favorites.size();
        vector<int> indegree(n);
        for (auto i: favorites) {
            ++indegree[i];
        }
        vector<bool> visited(n, false);
        vector<int> chains(n, 0);
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (indegree[i] == 0) {
                q.emplace(i);
            }
        }
        while (q.size()) {
            int front = q.front();
            q.pop();
            int next = favorites[front];
            visited[front] = true;
            chains[next] = chains[front] + 1;
            if (--indegree[next] == 0) {
                q.push(next);
            }
        }
        int maxCycle = 0;
        int totalChains = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                int current = i, cycleLength = 0;
                while (!visited[current]) {
                    visited[current] = true;
                    current = favorites[current];
                    cycleLength++;
                }
                if (cycleLength == 2) {
                    totalChains += 2 + chains[i] + chains[favorites[i]];
                } else {
                    maxCycle = max(maxCycle, cycleLength);
                }
            }
        }
        return max(maxCycle, totalChains);
    }

    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>> &prerequisites, vector<vector<int>> &queries) {
        vector<vector<bool>> graph(numCourses, vector<bool>(numCourses));
        for (auto &i: prerequisites) {
            graph[i[0]][i[1]] = true;
        }
        bool flag = true;
        while (flag) {
            flag = false;
            for (int i = 0; i < numCourses; ++i) {
                for (int j = 0; j < numCourses; ++j) {
                    if (i == j || graph[i][j]) continue;
                    for (int k = 0; k < numCourses; ++k) {
                        if (k == i || k == j) continue;
                        if (graph[i][k] && graph[k][j]) {
                            graph[i][j] = true;
                            flag = true;
                        }
                    }
                }
            }
        }
        vector<bool> res(queries.size());
        for (int i = 0; i < queries.size(); ++i) {
            res[i] = graph[queries[i][0]][queries[i][1]];
        }
        return res;
    }

    int findMaxFish(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.front().size();
        int res = 0;
        int sum = 0;
        vector<vector<bool>> visited(n, vector<bool>(m, false));
        auto dfs = [&](auto &&dfs, int x, int y) -> void {
            visited[x][y] = true;
            res = max(res, sum);
            if (grid[x][y] == 0)
                return;
            sum += grid[x][y];
            res = max(res, sum);
            constexpr int dx[4] = {0, 0, 1, -1};
            constexpr int dy[4] = {1, -1, 0, 0};
            for (int k = 0; k < 4; ++k) {
                int xx = x + dx[k];
                int yy = y + dy[k];
                if (xx < 0 || yy < 0 || xx >= n || yy >= m) continue;
                if (!visited[xx][yy]) {
                    dfs(dfs, xx, yy);
                }
            }
        };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (!visited[i][j]) {
                    sum = 0;
                    dfs(dfs, i, j);
                }
            }
        }
        return res;
    }

    vector<int> findRedundantConnection(vector<vector<int>> &edges) {
        int n = edges.size();
        vector<int> parent(n + 1);
        vector<int> size(n + 1, 1);
        for (int i = 0; i < n + 1; ++i) {
            parent[i] = i;
        }
        auto uf_find = [&](int index) {
            int next = parent[index];
            while (next != parent[next]) {
                next = parent[next];
            }
            return next;
        };
        auto uf_union = [&](int a, int b) {
            int pa = uf_find(a);
            int pb = uf_find(b);
            if (size[pa] > size[pb]) {
                parent[pb] = pa;
                size[pa] += size[pb];
            } else {
                parent[pa] = pb;
                size[pb] += size[pa];
            }
        };
        for (auto &i: edges) {
            int pa = uf_find(i[0]);
            int pb = uf_find(i[1]);
            if (pa == pb) return i;
            else
                uf_union(pa, pb);
        }
        return {};
    }

    bool isArraySpecial(vector<int> &nums) {
        for (auto &i: nums) {
            i &= 1;
        }
        for (int i = 0; i < nums.size() - 1; ++i) {
            if (!(nums[i] ^ nums[i + 1])) return false;
        }
        return true;
    }

    bool check(vector<int> &nums) {
        bool isOriginBegin = false;
        for (int i = 0; i < nums.size() - 1; ++i) {
            if (isOriginBegin) {
                if (nums[i + 1] > nums.front())
                    return false;
                if (nums[i + 1] < nums[i])
                    return false;
            } else {
                if (nums[i] <= nums[i + 1]) continue;
                else {
                    if (nums[i + 1] > nums.front()) return false;
                    isOriginBegin = true;
                }
            }
        }
        return true;
    }

    int longestMonotonicSubarray(vector<int> &nums) {
        int n = nums.size();
        if (n == 1) return 1;
        int res = (nums[0] == nums[1] ? 1 : 2);
        int op = (nums[0] > nums[1] ? -1 : (nums[0] == nums[1] ? 0 : 1));
        int len = (nums[0] == nums[1] ? 1 : 2);
        for (int i = 1; i < n - 1; ++i) {
            if (nums[i] < nums[i + 1]) {
                if (op == 1) {
                    res = max(res, ++len);
                } else {
                    res = max(res, len = 2);
                    op = 1;
                }
            } else if (nums[i] > nums[i + 1]) {
                if (op == -1) {
                    res = max(res, ++len);
                } else {
                    res = max(res, len = 2);
                    op = -1;
                }
            } else {
                op = 0;
                len = 1;
            }
        }
        return res;
    }

    int maxAscendingSum(vector<int> &nums) {
        int res = nums[0];
        int sum = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] > nums[i - 1]) {
                sum += nums[i];
            } else {
                sum = nums[i];
            }
            res = max(res, sum);
        }
        return res;
    }

    bool areAlmostEqual(const string &s1, const string &s2) {
        char s1c = 0;
        char s2c = 0;
        int n = s1.size();
        bool flag = false;
        for (int i = 0; i < n; ++i) {
            if (s1[i] == s2[i]) continue;
            else {
                if (flag) return false;
                if (s1c == 0) {
                    s1c = s1[i];
                    s2c = s2[i];
                } else {
                    if (s1c == s2[i] && s2c == s1[i]) {
                        flag = true;
                    } else
                        return false;
                }
            }
        }
        return s1c == 0 || flag;
    }

    int tupleSameProduct(vector<int> &nums) {
        int n = nums.size();
        int res = 0;
        unordered_map<int, int> times;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                res += times[nums[i] * nums[j]]++;
            }
        }
        return 8 * res;
    }

    vector<int> queryResults(int limit, vector<vector<int>> &queries) {
        vector<int> res(queries.size());
        unordered_map<int, int> colors;
        unordered_map<int, int> balls;
        int kinds = 0;
        for (int i = 0; i < queries.size(); ++i) {
            auto ball = queries[i][0];
            auto newColor = queries[i][1];
            if (colors.contains(ball)) {
                auto oldColor = colors[ball];
                if (--balls[oldColor] == 0)
                    --kinds;
            }
            if (++balls[newColor] == 1)
                ++kinds;
            colors[ball] = newColor;
            res[i] = kinds;
        }
        return res;
    }

    vector<int> lexicographicallySmallestArray(vector<int> &nums, int limit) {
        int n = nums.size();
        using vi = pair<int, int>;
        vector<vi> vis;
        vis.reserve(n);
        for (int i = 0; i < n; ++i) {
            vis.emplace_back(nums[i], i);
        }
        std::sort(vis.begin(), vis.end());
        vector<vector<vi>> groups;
        groups.push_back({vis[0]});
        for (int i = 1; i < n; ++i) {
            if (vis[i].first - vis[i - 1].first <= limit) {
                groups.back().emplace_back(vis[i]);
            } else {
                groups.push_back({vis[i]});
            }
        }
        for (int i = 0; i < groups.size(); ++i) {
            vector<int> indices;
            indices.reserve(groups[i].size());
            for (auto [_, i]: groups[i]) {
                indices.emplace_back(i);
            }
            std::sort(indices.begin(), indices.end());
            for (int k = 0; k < indices.size(); ++k) {
                nums[indices[k]] = groups[i][k].first;
            }
        }
        return nums;
    }

    string clearDigits(const string &s) {
        vector<char> st;
        st.reserve(s.size());
        for (auto i: s) {
            if (isdigit(i)) {
                if (st.size())
                    st.pop_back();
            } else {
                st.push_back(i);
            }
        }
        return {st.begin(), st.end()};
    }

    int largestIsland(vector<vector<int>> &grid) {
        constexpr int dx[4] = {0, 0, 1, -1};
        constexpr int dy[4] = {1, -1, 0, 0};
        int n = grid.size();
        int res = 0;
        vector<vector<int>> groupID(n, vector<int>(n, 0));
        unordered_map<int, int> groupSizes;
        auto dfs = [&](auto &&dfs, int x, int y) -> void {
            for (int k = 0; k < 4; ++k) {
                int xx = x + dx[k];
                int yy = y + dy[k];
                if (xx >= n || yy >= n || xx < 0 || yy < 0)
                    continue;
                if (groupID[xx][yy] != 0)
                    continue;
                if (grid[xx][yy] == 0)
                    continue;
                groupID[xx][yy] = groupID[x][y];
                res = max(res, ++groupSizes[groupID[x][y]]);
                dfs(dfs, xx, yy);
            }
        };
        int groupIndex = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1 && groupID[i][j] == 0) {
                    groupID[i][j] = groupIndex++;
                    res = max(res, ++groupSizes[groupID[i][j]]);
                    dfs(dfs, i, j);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0) {
                    int newSize = 1;
                    set<int> groups;
                    for (int k = 0; k < 4; ++k) {
                        int xx = i + dx[k];
                        int yy = j + dy[k];
                        if (xx >= n || yy >= n || xx < 0 || yy < 0)
                            continue;
                        if (grid[xx][yy] == 0)
                            continue;
                        if (groups.contains(groupID[xx][yy]))
                            continue;
                        newSize += groupSizes[groupID[xx][yy]];
                        groups.emplace(groupID[xx][yy]);
                    }
                    res = max(res, newSize);
                }
            }
        }
        return res;
    }

    string removeOccurrences(const string &s, const string &part) {
        string res;
        for (auto c: s) {
            res.push_back(c);
            if (c == part.back() && res.size() >= part.size()) {
                bool flag = true;
                for (auto partIter = part.crbegin(), resIter = res.crbegin(); flag && partIter != part.crend(); ++partIter, ++resIter) {
                    if (*partIter != *resIter) {
                        flag = false;
                    }
                }
                if (flag) {
                    for (auto _: part) {
                        res.pop_back();
                    }
                }
            }
        }
        return res;
    }

    // At any point, if the next item in the array is greater than (sum of all previous elements plus 1)
    // then add (sum of all previous elements plus 1) in the array.
    int minPatches(vector<int> &nums, int n) {
        long long presum = 0;
        int res = 0;
        int index = 0;
        while (presum < n) {
            if (index >= nums.size() || presum + 1 < nums[index]) {
                ++res;
                presum += presum + 1;
            } else {
                presum += nums[index];
                ++index;
            }
        }
        return res;
    }

    // 2493. Divide Nodes Into the Maximum Number of Groups
    int magnificentSets(int n, vector<vector<int>> &edges) {
        vector<vector<int>> graph(n + 1);
        vector<int> visited(n + 1, -1);
        for (const auto &edge: edges) {
            graph[edge[0]].push_back(edge[1]);
            graph[edge[1]].push_back(edge[0]);
        }
        int maxGroups = 0;
        unordered_set<int> componentNodes;
        auto findConnectedComponent = [&](auto &&findConnectedComponent, int node) -> void {
            componentNodes.insert(node);
            for (int neighbor: graph[node]) {
                if (!componentNodes.contains(neighbor)) {
                    findConnectedComponent(findConnectedComponent, neighbor);
                }
            }
        };
        auto getMaxDepth = [&](int start) {
            for (int node: componentNodes) visited[node] = -1;
            queue<int> q;
            int depth = 1;
            q.push(start);
            visited[start] = 1;
            while (!q.empty()) {
                int cur = q.front();
                q.pop();
                for (int neighbor: graph[cur]) {
                    if (visited[neighbor] == -1) {
                        visited[neighbor] = visited[cur] + 1;
                        depth = max(depth, visited[neighbor]);
                        q.push(neighbor);
                    } else if (abs(visited[cur] - visited[neighbor]) != 1) {
                        return -1;
                    }
                }
            }
            return depth;
        };
        for (int i = 1; i <= n; i++) {
            if (visited[i] != -1) continue;
            componentNodes.clear();
            findConnectedComponent(findConnectedComponent, i);
            int maxDepth = -1;
            for (int node: componentNodes) {
                maxDepth = max(maxDepth, getMaxDepth(node));
            }
            if (maxDepth == -1) return -1;
            maxGroups += maxDepth;
        }
        return maxGroups;
    }

    int maximumSum(vector<int> &nums) {
        auto calSumDigit = [](int num) {
            int res = 0;
            while (num >= 1) {
                res += (num % 10);
                num /= 10;
            }
            return res;
        };
        int res = -1;
        vector<int> digitSumMaxNum(100, -1);
        for (auto i: nums) {
            int digitSum = calSumDigit(i);
            if (digitSumMaxNum[digitSum] != -1) {
                res = max(res, i + digitSumMaxNum[digitSum]);
                digitSumMaxNum[digitSum] = max(digitSumMaxNum[digitSum], i);
            } else
                digitSumMaxNum[digitSum] = i;
        }
        return res;
    }

    int minOperations(vector<int> &nums, int k) {
        priority_queue<int, vector<int>, greater<>> pq(nums.begin(), nums.end());
        int res = 0;
        while (pq.size() >= 2 && pq.top() < k) {
            long long x = pq.top();
            pq.pop();
            long long y = pq.top();
            pq.pop();
            long long newNum = min(x, y) * 2 + max(x, y);
            if (newNum < k)
                pq.emplace(newNum);
            else
                pq.emplace(k);
            ++res;
        }
        return res;
    }

    int punishmentNumber(int n) {
        auto dfs = [&](auto &&dfs, long long int num, long long int target) -> bool {
            if (num == target) return true;
            for (int i = 10; i < num; i *= 10) {
                if (num / i + num % i == target) return true;
                if (target - num % i > 0 && dfs(dfs, num / i, target - num % i))
                    return true;
                if (target - num / i > 0 && dfs(dfs, num % i, target - num / i))
                    return true;
            }
            return false;
        };
        auto f = [&](long long num) -> bool {
            return dfs(dfs, num * num, num);
        };
        int res = 0;
        for (int i = 1; i <= n; ++i) {
            if (f(i)) res += i * i;
        }
        return res;
    }

    vector<int> constructDistancedSequence(int n) {
        int len = n * 2 - 1;
        vector<int> res(len, 0);
        vector<bool> placed(n + 1, false);
        auto dfs = [&](auto &&dfs, int i) -> bool {
            if (i >= len) return true;
            if (res[i] != 0) return dfs(dfs, i + 1);
            for (int k = n; k >= 1; --k) {
                if (placed[k]) continue;
                if (k == 1) {
                    res[i] = k;
                    placed[k] = true;
                    if (dfs(dfs, i + 1)) return true;
                    res[i] = 0;
                    placed[k] = false;

                } else if (i + k < len && res[i] == 0 && res[i + k] == 0) {
                    res[i] = res[i + k] = k;
                    placed[k] = true;
                    if (dfs(dfs, i + 1)) return true;
                    placed[k] = false;
                    res[i] = res[i + k] = 0;
                }
            }
            return false;
        };
        dfs(dfs, 0);
        return res;
    }
};

int main() {
    return 0;
}