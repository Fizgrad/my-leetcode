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
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <queue>
#include <ranges>
#include <ratio>
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

    int numTilePossibilities(const std::string &tiles) {
        int n = tiles.length();
        constexpr int SIZE = 'Z' - 'A' + 1;
        std::vector<int> counts(SIZE, 0);
        std::vector<int> fac(n + 1, 1);
        for (int i = 1; i <= n; i++) {
            fac[i] = i * fac[i - 1];
        }
        for (char c: tiles) {
            counts[c - 'A']++;
        }
        std::vector<int> lengthcounts(n + 1, 0);
        lengthcounts[0] = 1;
        for (int i = 0; i < SIZE; i++) {
            if (counts[i] > 0) {
                std::vector<int> temp(n + 1, 0);
                for (int j = 0; j <= n && lengthcounts[j] > 0; j++) {
                    for (int k = 1; k <= counts[i]; k++) {
                        int totallength = j + k;
                        temp[totallength] += lengthcounts[j] * fac[totallength] / (fac[k] * fac[j]);
                    }
                }
                for (int j = 0; j <= n; j++) {
                    lengthcounts[j] += temp[j];
                }
            }
        }
        return std::accumulate(lengthcounts.begin() + 1, lengthcounts.end(), 0);
    }

    string smallestNumber(const string &pattern) {
        string res;
        vector<bool> used(10, false);
        auto push = [&](char i) {
            used[i - '0'] = true;
            res.push_back(i);
        };
        auto pop = [&](char i) {
            res.pop_back();
            used[i - '0'] = false;
        };
        auto dfs = [&](auto &&dfs, int index) -> bool {
            if (index >= pattern.size()) return true;
            auto last = res.back();
            if (pattern[index] == 'I') {
                for (auto i = last + 1; i <= '9'; ++i) {
                    if (used[i - '0']) continue;
                    push(i);
                    if (dfs(dfs, index + 1))
                        return true;
                    pop(i);
                }
            } else {
                for (auto i = '1'; i < last; ++i) {
                    if (used[i - '0']) continue;
                    push(i);
                    if (dfs(dfs, index + 1))
                        return true;
                    pop(i);
                }
            }
            return false;
        };
        for (auto i = '1'; i <= '9'; ++i) {
            push(i);
            if (dfs(dfs, 0))
                return res;
            pop(i);
        }
        return res;
    }

    string getHappyString(int n, int k) {
        int nums = 3 * (1 << (n - 1));
        if (k > nums) return "";
        std::bitset<32> bits(k - 1);
        string res;
        res.push_back('a');
        if (bits.test(n))
            res.back() += 'c' - 'a';
        else if (bits.test(n - 1))
            res.back() += 'b' - 'a';
        for (int i = n - 2; i >= 0; --i) {
            auto c = res.back();
            auto append = 'a';
            if (append == c)
                ++append;
            while (bits.test(i)) {
                if (append != c)
                    bits.flip(i);
                ++append;
            }
            if (append == c)
                ++append;
            res.push_back(append);
        }
        return res;
    }

    string findDifferentBinaryString(vector<string> &nums) {
        // Because there are only n strings of length n,
        // ensuring that each string differs by at least one bit
        // will guarantee uniqueness.
        int n = nums.size();
        string res(n, '0');
        for (int i = 0; i < n; ++i) {
            res[i] = (res[i] == nums[i][i]) ? '1' : '0';
        }
        return res;
    }

    TreeNode *recoverFromPreorder(const string &traversal) {
        TreeNode *virtualNode = new TreeNode();
        int index = 0;
        int n = traversal.size();
        int level = 0;
        auto nextInt = [&]() -> int {
            int lastDigit = index;
            for (int i = index + 1; i < n; ++i) {
                if (std::isdigit(traversal[i])) {
                    lastDigit = i;
                } else
                    break;
            }
            int res = stoi(traversal.substr(index, lastDigit - index + 1));
            index = lastDigit + 1;
            return res;
        };

        auto numDash = [&]() -> int {
            if (index >= n) return -1;
            int res = 0;
            for (int i = index; i < n; ++i) {
                if (traversal[i] == '-') ++res;
                else
                    break;
            }
            index += res;
            return res;
        };

        auto preorder = [&](auto &&preorder, TreeNode *par, int depth) -> void {
            while (index < n && level == depth) {
                int num = nextInt();
                auto node = new TreeNode(num);
                if (par->left == nullptr) {
                    par->left = node;
                } else {
                    par->right = node;
                }
                level = numDash();
                if (level == depth + 1) {
                    preorder(preorder, node, depth + 1);
                }
            }
        };
        preorder(preorder, virtualNode, 0);
        return virtualNode->left;
    }

    TreeNode *constructFromPrePost(vector<int> &preorder, vector<int> &postorder) {
        using viIter = decltype(preorder.begin());
        auto f = [](auto &&f, viIter preorderBegin, viIter preorderEnd, viIter postorderBegin, viIter postorderEnd) -> TreeNode * {
            if (preorderBegin == preorderEnd || postorderBegin == postorderEnd) return nullptr;
            auto size = preorderEnd - preorderBegin;
            auto rootNode = new TreeNode(*preorderBegin);
            if (size == 1) return rootNode;
            int leftNum = *(preorderBegin + 1);
            int rightNum = *(postorderBegin + size - 2);
            if (leftNum == rightNum) {
                rootNode->left = f(f, preorderBegin + 1, preorderEnd, postorderBegin, postorderEnd - 1);
            } else {
                auto newPostorderEnd = std::find(postorderBegin, postorderEnd, leftNum);
                auto newPreorderEnd = std::find(preorderBegin, preorderEnd, rightNum);
                rootNode->left = f(f, preorderBegin + 1, newPreorderEnd, postorderBegin, newPostorderEnd + 1);
                rootNode->right = f(f, newPreorderEnd, preorderEnd, newPostorderEnd + 1, postorderEnd);
            }
            return rootNode;
        };
        return f(f, preorder.begin(), preorder.end(), postorder.begin(), postorder.end());
    }

    int mostProfitablePath(vector<vector<int>> &edges, int bob, vector<int> &amount) {
        int n = edges.size() + 1;
        vector<vector<int>> adjacentList(n);
        vector<vector<int>> treeNext(n);
        vector<bool> visited(n, false);
        for (auto &i: edges) {
            adjacentList[i[0]].emplace_back(i[1]);
            adjacentList[i[1]].emplace_back(i[0]);
        }
        vector<int> bobPath;
        auto constructTree = [&](auto &&constructTree, int node) -> bool {
            bool flag = false;
            for (auto next: adjacentList[node]) {
                if (visited[next]) continue;
                visited[next] = true;
                treeNext[node].emplace_back(next);
                flag = constructTree(constructTree, next) || flag;// note: short circuit!
            }
            flag = (node == bob || flag);
            if (flag)
                bobPath.emplace_back(node);
            return flag;
        };
        visited[0] = true;
        constructTree(constructTree, 0);

        auto bfs = [&](auto &&bfs, int time, int alice) -> int {
            if (treeNext[alice].empty()) return amount[alice];
            int res = std::numeric_limits<int>::min();
            if (time >= bobPath.size()) {
                for (auto next: treeNext[alice]) {
                    res = max(res, amount[alice] + bfs(bfs, time + 1, next));
                }
            } else {
                int bob = bobPath[time];
                auto back = amount[bob];
                if (alice == bob) {
                    amount[alice] = (amount[alice] >> 1);
                } else {
                    amount[bob] = 0;
                }
                for (auto next: treeNext[alice]) {
                    res = max(res, amount[alice] + bfs(bfs, time + 1, next));
                }
                amount[bob] = back;
            }
            return res;
        };
        return bfs(bfs, 0, 0);
    }

    int numOfSubarrays(vector<int> &arr) {
        int n = arr.size();
        int prefixSum = 0;
        int res = 0;
        constexpr int MOD = 1e9 + 7;
        long long odd = 0;
        long long even = 1;
        for (int i = 0; i < n; ++i) {
            prefixSum += arr[i];
            if (prefixSum & 1) {
                res = (res + even) % MOD;
                ++odd;
            } else {
                res = (res + odd) % MOD;
                ++even;
            }
        }
        return res;
    }

    int maxAbsoluteSum(vector<int> &nums) {
        int n = nums.size();
        int minPrefixSum = std::numeric_limits<int>::max();
        int maxPrefixSum = std::numeric_limits<int>::min();
        int res = 0;
        int prefixSum = 0;
        for (auto i: nums) {
            prefixSum += i;
            res = std::max(res, std::abs(prefixSum));
            minPrefixSum = std::min(minPrefixSum, prefixSum);
            maxPrefixSum = std::max(maxPrefixSum, prefixSum);
            res = std::max(res, std::abs(prefixSum - maxPrefixSum));
            res = std::max(res, std::abs(prefixSum - minPrefixSum));
        }
        return res;
    }

    int lenLongestFibSubseq(vector<int> &arr) {
        int n = arr.size();
        int res = 0;
        auto dfs = [&](auto &&dfs, int a, int b, int len) -> void {
            res = max(res, len);
            auto iter = lower_bound(arr.begin(), arr.end(), a + b);
            if (iter == arr.end() || *iter != a + b) return;
            dfs(dfs, b, a + b, len + 1);
        };
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                dfs(dfs, arr[i], arr[j], 2);
            }
        }
        return res >= 3 ? res : 0;
    }

    string shortestCommonSupersequence(string str1, string str2) {
        int n = str1.size();
        int m = str2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
                if (str1[i] == str2[j]) {
                    dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + 1);
                }
            }
        }
        string longestCommonSubsequence;
        int x = n;
        int y = m;
        while (x > 0 && y > 0) {
            if (dp[x][y] > dp[x][y - 1] && dp[x][y] > dp[x - 1][y]) {
                longestCommonSubsequence.push_back(str1[x - 1]);
                --x;
                --y;
            } else if (dp[x][y] == dp[x][y - 1]) {
                --y;
            } else if (dp[x][y] == dp[x - 1][y])
                --x;
        }
        string res;
        int index1 = 0;
        int index2 = 0;
        int indexCommonSubsequence = longestCommonSubsequence.size() - 1;
        while (index1 < n || index2 < m) {
            if (indexCommonSubsequence >= 0) {
                if (longestCommonSubsequence[indexCommonSubsequence] == str1[index1] && str1[index1] == str2[index2]) {
                    res.push_back(str1[index1]);
                    ++index1;
                    ++index2;
                    --indexCommonSubsequence;
                } else if (longestCommonSubsequence[indexCommonSubsequence] == str1[index1]) {
                    res.push_back(str2[index2]);
                    ++index2;
                } else if (longestCommonSubsequence[indexCommonSubsequence] == str2[index2]) {
                    res.push_back(str1[index1]);
                    ++index1;
                } else {
                    res.push_back(str1[index1]);
                    res.push_back(str2[index2]);
                    ++index1;
                    ++index2;
                }
            } else {
                if (index1 >= n) {
                    res.push_back(str2[index2]);
                    ++index2;
                } else if (index2 >= m) {
                    res.push_back(str1[index1]);
                    ++index1;
                } else {
                    res.push_back(str1[index1]);
                    res.push_back(str2[index2]);
                    ++index1;
                    ++index2;
                }
            }
        }
        return res;
    }

    vector<int> applyOperations(vector<int> &nums) {
        int n = nums.size();
        for (int i = 1; i < n; ++i) {
            if (nums[i] == nums[i - 1]) {
                nums[i - 1] *= 2;
                nums[i] = 0;
            }
        }
        int index = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] != 0) {
                std::swap(nums[i], nums[index++]);
            }
        }
        return nums;
    }

    vector<vector<int>> mergeArrays(vector<vector<int>> &nums1, vector<vector<int>> &nums2) {
        int index1 = 0;
        int index2 = 0;
        int n = nums1.size();
        int m = nums2.size();
        vector<vector<int>> res;
        while (index1 < n || index2 < m) {
            if (index1 >= n) {
                res.push_back(nums2[index2++]);
            } else if (index2 >= m) {
                res.push_back(nums1[index1++]);
            } else {
                if (nums1[index1][0] == nums2[index2][0]) {
                    res.push_back({nums1[index1][0], nums1[index1][1] + nums2[index2][1]});
                    ++index1;
                    ++index2;
                } else if (nums1[index1][0] > nums2[index2][0]) {
                    res.push_back(nums2[index2++]);
                } else {
                    res.push_back(nums1[index1++]);
                }
            }
        }
        return res;
    }

    vector<int> pivotArray(vector<int> &nums, int pivot) {
        vector<vector<int>> index(3, vector<int>());
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] == pivot) {
                index[1].push_back(i);
            } else if (nums[i] < pivot) {
                index[0].push_back(i);
            } else {
                index[2].push_back(i);
            }
        }
        vector<int> res;
        res.reserve(nums.size());
        for (auto &arr: index) {
            for (auto i: arr) {
                res.push_back(nums[i]);
            }
        }
        return res;
    }

    bool checkPowersOfThree(int n) {
        int temp = n;
        while (temp > 1) {
            if (temp % 3 == 0) {
                temp /= 3;
            } else {
                if ((temp - 1) % 3 == 0) {
                    temp = (temp - 1) / 3;
                } else
                    return false;
            }
        }
        return true;
    }

    long long coloredCells(int n) {
        auto f = [](long long n) -> long long {
            return n * n;
        };
        return f(n) + f(n - 1);
    }

    vector<int> findMissingAndRepeatedValues(vector<vector<int>> &grid) {
        int xorSum = 0;
        int n = grid.size();
        std::bitset<2501> visited;
        int repeat = -1;
        for (auto &arr: grid) {
            for (auto i: arr) {
                xorSum ^= i;
                if (visited.test(i)) {
                    repeat = i;
                } else
                    visited.set(i);
            }
        }
        for (int i = 1; i <= n * n; ++i) {
            xorSum ^= i;
        }
        return {repeat, xorSum ^ repeat};
    }

    vector<int> closestPrimes(int left, int right) {
        constexpr int SIZE = 1e6 + 1;
        std::bitset<SIZE> isPrime;
        isPrime.set(1);
        for (int i = 2; i < SIZE; ++i) {
            if (isPrime.test(i))
                continue;
            for (int k = 2; k * i < SIZE; ++k) {
                isPrime.set(k * i);
            }
        }
        int res = right - left + 1;
        vector<int> resArr = {-1, -1};
        for (int i = left; i <= right;) {
            int j = i + 1;
            if (isPrime.test(i)) {
                i = j;
                continue;
            }
            for (; j <= right; ++j) {
                if (isPrime.test(j))
                    continue;
                if (j - i < res) {
                    res = min(res, j - i);
                    resArr = {i, j};
                }
                break;
            }
            i = j;
        }
        return resArr;
    }

    int minimumRecolors(const string &blocks, int k) {
        const int n = blocks.size();
        int whiteNums = std::count(blocks.begin(), blocks.begin() + k, 'W');
        int res = whiteNums;
        for (int l = 0, r = k; r < n; r++, l++) {
            whiteNums += (blocks[r] == 'W') - (blocks[l] == 'W');
            res = min(res, whiteNums);
        }
        return res;
    }

    int numberOfAlternatingGroups(vector<int> &colors, int k) {
        int n = colors.size();
        int equal = 0;
        int res = 0;

        auto get = [&](int index) {
            return colors[(index + n) % n];
        };

        for (int i = 1; i < k; ++i) {
            if (get(i) == get(i - 1)) ++equal;
        }
        if (equal == 0) ++res;
        for (int i = 1; i < n; ++i) {
            if (get(i) == get(i - 1)) --equal;
            if (get(i + k - 1) == get(i + k - 2)) ++equal;
            if (equal == 0) ++res;
        }
        return res;
    }

    long long countOfSubstrings(const string &word, int k) {
        long long res = 0;
        int n = word.size();
        auto vowelIndex = [&](char c) {
            switch (c) {
                case 'a':
                    return 1;
                case 'e':
                    return 2;
                case 'i':
                    return 3;
                case 'o':
                    return 4;
                case 'u':
                    return 5;
                default:
                    return 0;
            }
        };
        vector<int> times(6, 0);
        vector<vector<int>> indexTimes(n + 1, vector<int>(6, 0));
        for (int i = 0; i < n; ++i) {
            ++times[vowelIndex(word[i])];
            indexTimes[i + 1] = times;
        }
        for (int i = 0; i < n; ++i) {
            if (indexTimes[i + 1][0] < k) {
                continue;
            }
            if (std::any_of(indexTimes[i + 1].begin() + 1, indexTimes[i + 1].end(), [](auto c) {
                    return c < 1;
                })) {
                continue;
            }
            int left = 0;
            int right = i + 1;
            int exactK = -1;
            while (left <= right) {
                int mid = (left + right) >> 1;
                if (indexTimes[i + 1][0] - indexTimes[mid][0] < k) {
                    right = mid - 1;
                } else if (indexTimes[i + 1][0] == k + indexTimes[mid][0]) {
                    auto isOK = [&]() {
                        for (int k = 1; k < 6; ++k) {
                            if (indexTimes[i + 1][k] - indexTimes[mid][k] < 1)
                                return false;
                        }
                        return true;
                    };
                    if (isOK()) {
                        exactK = mid;
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                } else {
                    left = mid + 1;
                }
            }
            if (exactK == -1) continue;
            else {
                int left = 0;
                int right = exactK + 1;
                int beginK = exactK;
                while (left <= right) {
                    int mid = (left + right) >> 1;
                    if (indexTimes[i + 1][0] - indexTimes[mid][0] < k) {
                        right = mid - 1;
                    } else if (indexTimes[i + 1][0] == k + indexTimes[mid][0]) {
                        beginK = mid;
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                }
                res += exactK - beginK + 1;
            }
        }
        return res;
    }

    int numberOfSubstrings(const string &s) {
        int i = 0;
        int j = 0;
        vector<int> times(3, 0);
        int res = 0;
        while (j < s.size()) {
            ++times[s[j] - 'a'];
            while (std::all_of(times.begin(), times.end(), [](auto i) {
                return i >= 1;
            })) {
                --times[s[i] - 'a'];
                ++i;
            }
            res += i;
            ++j;
        }
        return res;
    }

    int maximumCount(vector<int> &nums) {
        return std::max(
                std::ranges::count_if(nums, [](int x) { return x > 0; }),
                std::ranges::count_if(nums, [](int x) { return x < 0; }));
    }

    // 3356. Zero Array Transformation II
    // difference array
    int minZeroArray(vector<int> &nums, vector<vector<int>> &queries) {
        int n = nums.size();
        vector<int> diffArray(n + 1, 0);
        int sum = 0;
        int queriesIndex = 0;
        for (int i = 0; i < n; ++i) {
            sum += diffArray[i];
            while (nums[i] + sum > 0) {
                if (queriesIndex >= queries.size()) return -1;
                int left = queries[queriesIndex][0];
                int right = queries[queriesIndex][1];
                int value = queries[queriesIndex][2];
                diffArray[left] -= value;
                diffArray[right + 1] += value;
                if (i >= left && i <= right) {
                    sum -= value;
                }
                ++queriesIndex;
            }
        }
        return queriesIndex;
    }

    // 2226. Maximum Candies Allocated to K Children
    int maximumCandies(vector<int> &candies, long long k) {
        // add suffix 'l' to std::accumulate's parameter 'init'
        long long maxNum = std::accumulate(candies.begin(), candies.end(), 0l) / k;
        long long minNum = 0;
        int res = 0;
        while (maxNum >= minNum) {
            long long mid = (maxNum + minNum) >> 1;
            if ([&]() -> bool {
                    if (mid == 0) return true;
                    long long remains = k;
                    for (auto i: candies) {
                        remains -= (i / mid);
                        if (remains <= 0) return true;
                    }
                    return false;
                }()) {
                res = mid;
                minNum = mid + 1;
            } else
                maxNum = mid - 1;
        }
        return res;
    }

    int minCapability(vector<int> &nums, int k) {
        int n = nums.size();
        int res = -1;
        int right = std::ranges::max(nums);
        int left = std::ranges::min(nums);
        while (left <= right) {
            int mid = (right + left) >> 1;
            if ([&]() -> bool {
                    int robbed = 0;
                    for (int i = 0; i < n; ++i) {
                        if (nums[i] <= mid) {
                            ++robbed;
                            ++i;
                        }
                    }
                    return robbed >= k ? true : false;
                }()) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    long long repairCars(vector<int> &ranks, int cars) {
        long long res = -1;
        long long left = 1;
        long long right = static_cast<long long>(std::ranges::min(ranks)) * cars * cars;
        while (left <= right) {
            long long mid = (left + right) >> 1;
            if ([&]() -> bool {
                    int remains = cars;
                    for (auto i: ranks) {
                        remains -= static_cast<int>(sqrt(static_cast<double>(mid) / i));
                        if (remains <= 0) return true;
                    }
                    return false;
                }()) {
                res = mid;
                right = mid - 1;
            } else
                left = mid + 1;
        }
        return res;
    }

    bool divideArray(vector<int> &nums) {
        constexpr int SIZE = 500 + 5;
        vector<bool> times(SIZE, false);
        for (auto i: nums) {
            times[i] = !times[i];
        }
        return !std::ranges::any_of(times, [](auto i) -> bool { return i; });
    }

    int longestNiceSubarray(vector<int> &nums) {
        int n = nums.size();
        int res = 1;
        int andSum = 0;
        int i = 0;
        int j = 0;
        while (j < n) {
            while (nums[j] & andSum) {
                andSum = andSum & (~nums[i++]);
            }
            andSum |= nums[j];
            ++j;
            res = max(res, j - i);
        }
        return res;
    }

    vector<string> findAllRecipes(vector<string> &recipes, vector<vector<string>> &ingredients, vector<string> &supplies) {
        int n = recipes.size();
        vector<string> res;
        vector<int> indegree(n, 0);
        unordered_map<string, vector<int>> next;
        for (int i = 0; i < n; ++i) {
            indegree[i] = ingredients[i].size();
            for (auto &s: ingredients[i]) {
                next[s].emplace_back(i);
            }
        }
        while (!supplies.empty()) {
            auto i = supplies.back();
            supplies.pop_back();
            for (auto &k: next[i]) {
                --indegree[k];
                if (indegree[k] == 0) {
                    supplies.emplace_back(recipes[k]);
                    res.emplace_back(recipes[k]);
                }
            }
        }
        return res;
    }

    int countCompleteComponents(int n, vector<vector<int>> &edges) {
        vector<int> parents(n, 0);
        std::iota(parents.begin(), parents.end(), 0);
        auto uf_find = [&](int node) {
            int iter = node;
            while (iter != parents[iter]) {
                iter = parents[iter];
            }
            while (node != iter) {
                auto next = parents[node];
                parents[node] = iter;
                node = next;
            }
            return iter;
        };

        auto uf_union = [&](int a, int b) {
            int pa = uf_find(a);
            int pb = uf_find(b);
            parents[pb] = pa;
        };

        for (auto &i: edges) {
            uf_union(i[0], i[1]);
        }

        std::unordered_map<int, int> vertexNum;
        std::unordered_map<int, int> edgeNum;
        for (int i = 0; i < n; ++i) {
            ++vertexNum[uf_find(i)];
        }
        for (auto &i: edges) {
            ++edgeNum[uf_find(i[0])];
        }
        int res = 0;
        for (auto &i: vertexNum) {
            if ((i.second * (i.second - 1) >> 1) == edgeNum[i.first]) ++res;
        }
        return res;
    }

    int countPaths(int n, vector<vector<int>> &roads) {
        using TimeNode = pair<long long, int>;
        priority_queue<TimeNode, deque<TimeNode>, greater<>> pq;
        vector<vector<TimeNode>> graph(n);
        vector<int> visitedNums(n, 0);
        vector<long long> visitedTimes(n, std::numeric_limits<long long>::max());
        for (auto &i: roads) {
            graph[i[0]].emplace_back(i[2], i[1]);
            graph[i[1]].emplace_back(i[2], i[0]);
        }
        pq.emplace(0, 0);
        visitedNums[0] = 1;
        visitedTimes[0] = 0;
        constexpr int MOD = 1e9 + 7;
        while (pq.size()) {
            auto top = pq.top();
            pq.pop();
            if (top.first > visitedTimes[top.second]) continue;
            for (auto &i: graph[top.second]) {
                auto newTime = i.first + top.first;
                if (newTime < visitedTimes[i.second]) {
                    pq.emplace(newTime, i.second);
                    visitedNums[i.second] = visitedNums[top.second];
                    visitedTimes[i.second] = newTime;
                } else if (newTime == visitedTimes[i.second]) {
                    visitedNums[i.second] = (visitedNums[i.second] + visitedNums[top.second]) % MOD;
                }
            }
        }
        return visitedNums[n - 1];
    }

    vector<int> partitionLabels(const string &s) {
        constexpr int SIZE = 'z' - 'a' + 1;
        vector<int> last(SIZE, std::numeric_limits<int>::min());
        for (int i = 0; i < s.size(); ++i) {
            last[s[i] - 'a'] = max(last[s[i] - 'a'], i);
        }
        vector<int> res;
        int end = 0;
        int len = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (i > end) {
                res.push_back(len);
                len = 0;
            }
            end = max(end, last[s[i] - 'a']);
            ++len;
        }
        res.push_back(len);
        return res;
    }

    long long putMarbles(vector<int> &weights, int k) {
        int n = weights.size();
        vector<long long> splits;
        for (int i = 1; i < weights.size(); ++i) {
            splits.emplace_back(weights[i - 1] + weights[i]);
        }
        std::sort(splits.begin(), splits.end());
        long long res = 0;
        for (int i = 0; i < k - 1; ++i) {
            res += splits[splits.size() - 1 - i] - splits[i];
        }
        return res;
    }

    long long mostPoints(vector<vector<int>> &questions) {
        vector<long long> dp(questions.size(), -1);
        for (int i = questions.size() - 1; i >= 0; --i) {
            dp[i] = questions[i][0];
            if (i + questions[i][1] + 1 < questions.size())
                dp[i] = max(dp[i], dp[i + questions[i][1] + 1] + questions[i][0]);
            if (i + 1 < questions.size()) dp[i] = max(dp[i], dp[i + 1]);
        }
        return dp[0];
    }

    long long maximumTripletValue(vector<int> &nums) {
        int n = nums.size();
        long long res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int z = j + 1; z < n; ++z) {
                    res = max(res, static_cast<long long>(nums[z]) * (nums[i] - nums[j]));
                }
            }
        }
        return res;
    }

    long long numberOfPowerfulInt(long long start, long long finish, long long limit, const string &s) {
        long long s_num = stoll(s);
        if (finish < s_num) return 0;
        long long start_biased = std::max(0ll, start - s_num);
        long long finish_biased = std::max(0ll, finish - s_num);
        int s_len = s.size();
        auto pow = [](long long a, long long b) {
            long long res = 1;
            if (b & 1) {
                res *= a;
            }
            b >>= 1;
            while (b >= 1) {
                a *= a;
                if (b & 1) {
                    res *= a;
                }
                b >>= 1;
            }
            return res;
        };
        long long factor = pow(10, s_len);
        long long finish_trimed = finish_biased / factor;
        long long start_trimed = (start_biased + factor - 1) / factor;
        if (start_trimed > finish_trimed) return 0;
        auto count = [&](auto &&self, long long x) -> long long {
            if (x < 0) return 0;
            vector<int> digits;
            {
                long long tmp = x;
                if (tmp == 0) {
                    digits.push_back(0);
                } else {
                    while (tmp > 0) {
                        digits.push_back(tmp % 10);
                        tmp /= 10;
                    }
                    reverse(digits.begin(), digits.end());
                }
            }

            vector<vector<long long>> dp(digits.size(), vector<long long>(2, -1));
            function<long long(int, bool)> dfs = [&](int pos, bool isTight) -> long long {
                if (pos == (int) digits.size()) {
                    return 1LL;
                }
                if (dp[pos][isTight ? 1 : 0] != -1) {
                    return dp[pos][isTight ? 1 : 0];
                }
                int up = isTight ? digits[pos] : limit;

                long long res = 0;
                for (int dig = 0; dig <= up; dig++) {
                    bool nextTight = (isTight && (dig == up));
                    if (dig <= limit) {
                        res += dfs(pos + 1, nextTight);
                    }
                }
                dp[pos][isTight ? 1 : 0] = res;
                return res;
            };
            return dfs(0, true);
        };
        return count(count, finish_trimed) - (start_trimed > 0 ? count(count, start_trimed - 1) : 0);
    }

    int countSymmetricIntegers(int low, int high) {
        int res = 0;
        if (low <= 99) {
            for (int i = max(10, low); i <= min(high, 99); ++i) {
                if (i / 10 == i % 10) ++res;
            }
        }
        if (high >= 1000) {
            vector<int> sum_of_digits(100, 0);
            for (int i = 0; i < sum_of_digits.size(); ++i) {
                sum_of_digits[i] = i / 10 + i % 10;
            }
            for (int i = max(1000, low); i <= min(9999, high); ++i) {
                if (sum_of_digits[i / 100] == sum_of_digits[i % 100]) ++res;
            }
        }
        return res;
    }

    int countGoodTriplets(vector<int> &arr, int a, int b, int c) {
        int res = 0;
        vector<int> indices;
        for (int i = 0; i < arr.size(); ++i) {
            indices.clear();
            for (int k = i + 1; k < arr.size(); ++k) {
                int diff = std::abs(arr[k] - arr[i]);
                if (diff <= c) {
                    for (auto j: indices) {
                        if (std::abs(arr[j] - arr[k]) <= b)
                            ++res;
                    }
                }
                if (diff <= a) {
                    indices.emplace_back(k);
                }
            }
        }
        return res;
    }

    long long countGood(vector<int> &nums, int k) {
        constexpr int N = 1e5 + 1;
        int n = nums.size();
        long long res = 0;
        int left = 0;
        unordered_map<int, int> times;
        for (int i = 0; i < n; ++i) {
            k -= times[nums[i]]++;
            while (k <= 0) { k += --times[nums[left++]]; }
            res += left;
        }
        return res;
    }

    int countPairs(vector<int> &nums, int k) {
        int n = nums.size();
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] != nums[j])
                    continue;
                if (i * j % k == 0)
                    ++res;
            }
        }
        return res;
    }

    string countAndSay(int n) {
        string res = "1";
        for (int i = 1; i < n; ++i) {
            char prev = res[0];
            string temp;
            int times = 1;
            for (int k = 1; k < res.size(); ++k) {
                if (res[k] == prev) {
                    ++times;
                } else {
                    temp.push_back('0' + times);
                    temp.push_back(prev);
                    times = 1;
                    prev = res[k];
                }
            }
            temp.push_back('0' + times);
            temp.push_back(prev);
            res.swap(temp);
        }
        return res;
    }

    long long countFairPairs(vector<int> &nums, int lower, int upper) {
        int n = nums.size();
        std::sort(nums.begin(), nums.end());
        long long res = 0;
        for (int i = 1; i < n; ++i) {
            auto _begin = nums.begin();
            auto _end = nums.begin() + i;
            auto lower_iter = std::lower_bound(_begin, _end, lower - nums[i]);
            auto upper_iter = std::upper_bound(_begin, _end, upper - nums[i]);
            res += upper_iter - lower_iter;
        }
        return res;
    }

    int numRabbits(vector<int> &answers) {
        unordered_map<int, int> remains;
        int res = 0;
        for (auto i: answers) {
            if (remains[i] == 0) {
                res += i + 1;
                remains[i] = i;
            } else {
                --remains[i];
            }
        }
        return res;
    }

    int countLargestGroup(int n) {
        vector<int> sizes(50, 0);
        for (int i = 1; i <= n; ++i) {
            if (i <= 9) {
                sizes[i]++;
            } else {
                int sum = 0;
                int iter = i;
                while (iter >= 1) {
                    sum += iter % 10;
                    iter /= 10;
                }
                sizes[sum]++;
            }
        }
        std::ranges::sort(sizes);
        int max = sizes.back();
        int res = 0;
        for (int i = sizes.size() - 1; i >= 0; --i) {
            if (sizes[i] == max) {
                ++res;
            }
        }
        return res;
    }

    int countCompleteSubarrays(vector<int> &nums) {
        constexpr int N = 2000 + 1;
        vector<int> indices(N, -1);
        int index = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (indices[nums[i]] == -1)
                indices[nums[i]] = index++;
            nums[i] = indices[nums[i]];
        }
        vector<int> times(index, 0);
        int left = 0;
        int right = 0;
        int res = 0;
        while (right < nums.size()) {
            ++times[nums[right]];
            if (std::ranges::all_of(times, [](auto element) {
                    return element > 0;
                })) {
                while (left < right && times[nums[left]] > 1) {
                    --times[nums[left]];
                    ++left;
                }
                res += left + 1;
            }
            ++right;
        }
        return res;
    }

    long long countInterestingSubarrays(vector<int> &nums, int modulo, int k) {
        int n = nums.size();
        long long res = 0;
        int prefix = 0;
        unordered_map<int, int> times;
        times[0]++;
        for (int i = 0; i < n; ++i) {
            prefix = (prefix + ((nums[i] % modulo == k) ? 1 : 0)) % modulo;
            res += times[(prefix - k + modulo) % modulo];
            ++times[prefix % modulo];
        }
        return res;
    }

    long long countSubarrays(vector<int> &nums, int minK, int maxK) {
        int max_idx = -1;
        int min_idx = -1;
        int bad_idx = -1;
        long long res = 0;
        for (int i = 0; i < nums.size(); ++i) {
            int num = nums[i];
            if (num < minK || num > maxK) bad_idx = i;
            if (minK == num) min_idx = i;
            if (maxK == num) max_idx = i;
            res += max(0, min(max_idx, min_idx) - bad_idx);
        }
        return res;
    }

    int countSubarrays(vector<int> &nums) {
        int res = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (i + 2 >= nums.size()) return res;
            if ((nums[i] + nums[i + 2]) * 2 == nums[i + 1]) ++res;
        }
        return res;
    }

    long long countSubarrays(vector<int> &nums, long long k) {
        long long res = 0;
        const int n = nums.size();
        int left = 0;
        long long sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
            while (left < i && sum * (i - left + 1) >= k) {
                sum -= nums[left++];
            }
            if (sum * (i - left + 1) < k) {
                res += i - left + 1;
            }
        }
        return res;
    }

    long long countSubarrays(vector<int> &nums, int k) {
        int maxNum = 0;
        int iterStart = 0;
        for (int i = nums.size() - 1; i >= 0; --i) {
            if (nums[i] >= maxNum) {
                iterStart = i;
                maxNum = nums[i];
            }
        }
        int startIndex = iterStart;
        int times = 0;
        long long res = 0;
        for (int i = iterStart; i < nums.size(); ++i) {
            if (nums[i] == maxNum) {
                ++times;
            }
            while (times > k) {
                if (nums[startIndex] == maxNum)
                    --times;
                ++startIndex;
            }
            while (times == k) {
                if (nums[startIndex] == maxNum) {
                    res += startIndex + 1;
                    break;
                }
                ++startIndex;
            }
        }
        return res;
    }

    int findNumbers(vector<int> &nums) {
        auto f = [](int num) -> bool {
            int temp = num;
            int len = 0;
            while (temp >= 1) {
                ++len;
                temp /= 10;
            }
            return (len & 1);
        };
        int res = 0;
        for (auto i: nums) {
            if (!f(i))
                ++res;
        }
        return res;
    }

    int maxTaskAssign(vector<int> &tasks, vector<int> &workers, int pills, int strength) {
        int n = tasks.size();
        int m = workers.size();
        int res = 0;
        int left = 0;
        int right = min(n, m);
        std::ranges::sort(tasks);
        std::ranges::sort(workers);
        while (left <= right) {
            int mid = (left + right) >> 1;
            if ([&](int k) -> bool {
                    int taskID = 0;
                    int pillsRemain = pills;
                    deque<int> canCompleteTasks;
                    for (int workerID = m - k; workerID < m; ++workerID) {
                        while (taskID < k) {
                            if (tasks[taskID] <= workers[workerID] + strength) {
                                canCompleteTasks.emplace_back(taskID);
                                ++taskID;
                            } else
                                break;
                        }
                        if (canCompleteTasks.empty()) return false;
                        if (tasks[canCompleteTasks.front()] <= workers[workerID]) {
                            canCompleteTasks.pop_front();
                            continue;
                        }
                        if (tasks[canCompleteTasks.back()] <= workers[workerID] + strength) {
                            if (pillsRemain > 0) {
                                canCompleteTasks.pop_back();
                                --pillsRemain;
                            } else
                                return false;
                        }
                    }
                    return true;
                }(mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    string pushDominoes(const string &dominoes) {
        int n = dominoes.size();
        string res(dominoes);
        int i = 0;
        while (i < n) {
            auto c = res[i];
            if (c == 'L') {
                for (int k = i - 1; k >= 0; --k) {
                    if (res[k] == '.')
                        res[k] = 'L';
                    else
                        break;
                }
                ++i;
                continue;
            } else if (c == 'R') {
                int nextIndex = n;
                char nextState = ' ';
                for (int k = i + 1; k < n; ++k) {
                    if (res[k] == 'L') {
                        nextIndex = k;
                        nextState = 'L';
                        break;
                    } else if (res[k] == 'R') {
                        nextIndex = k;
                        nextState = 'R';
                        break;
                    }
                }
                if (nextIndex == n) {
                    for (int k = i + 1; k < n; ++k) {
                        res[k] = 'R';
                    }
                    break;
                } else {
                    if (nextState == 'R') {
                        for (int k = i + 1; k < nextIndex; ++k) {
                            res[k] = 'R';
                        }
                        i = nextIndex;
                        continue;
                    } else {
                        int rightIndex = nextIndex - 1;
                        int leftIndex = i + 1;
                        while (leftIndex < rightIndex) {
                            res[rightIndex] = 'L';
                            res[leftIndex] = 'R';
                            ++leftIndex;
                            --rightIndex;
                        }
                        i = nextIndex + 1;
                    }
                }
            } else
                ++i;
        }
        return res;
    }

    int minDominoRotations(vector<int> &tops, vector<int> &bottoms) {
        int n = tops.size();
        int a = tops.front();
        int b = bottoms.front();
        for (int i = 1; i < n; ++i) {
            if (tops[i] != a && bottoms[i] != a) {
                a = -1;
            }
            if (tops[i] != b && bottoms[i] != b) {
                b = -1;
            }
        }
        if (a == -1 && b == -1) return -1;
        int res = n;
        if (a != -1) {
            int temp = 0;
            for (int i = 0; i < n; ++i) {
                temp += (tops[i] == a ? 0 : 1);
            }
            res = min(temp, res);
            temp = 0;
            for (int i = 0; i < n; ++i) {
                temp += (bottoms[i] == a ? 0 : 1);
            }
            res = min(temp, res);
        }
        if (b != -1) {
            int temp = 0;
            for (int i = 0; i < n; ++i) {
                temp += (tops[i] == b ? 0 : 1);
            }
            res = min(temp, res);
            temp = 0;
            for (int i = 0; i < n; ++i) {
                temp += (bottoms[i] == b ? 0 : 1);
            }
            res = min(temp, res);
        }
        return res;
    }

    int numEquivDominoPairs(vector<vector<int>> &dominoes) {
        map<pair<int, int>, int> nums;
        int res = 0;
        for (auto &i: dominoes) {
            res += nums[std::make_pair(max(i[0], i[1]), min(i[0], i[1]))]++;
        }
        return res;
    }

    vector<int> buildArray(vector<int> &nums) {
        vector<int> res(nums.size());
        for (int i = 0; i < nums.size(); ++i) {
            res[i] = nums[nums[i]];
        }
        return res;
    }

    int minTimeToReach(vector<vector<int>> &moveTime) {
        using TimeCoordinate = pair<int, pair<int, int>>;
        priority_queue<TimeCoordinate, vector<TimeCoordinate>, greater<>> pq;
        int n = moveTime.size();
        int m = moveTime.front().size();
        constexpr int dd[5] = {0, 1, 0, -1, 0};
        pq.emplace(std::make_pair(0, std::make_pair(0, 0)));
        vector<vector<bool>> visited(n, vector<bool>(m, false));
        while (pq.size()) {
            auto [time, coordinate] = pq.top();
            auto [x, y] = coordinate;
            pq.pop();
            if (x == n - 1 && y == m - 1) return time;
            if (visited[x][y]) continue;
            visited[x][y] = true;
            for (int i = 0; i < 4; ++i) {
                int xx = x + dd[i];
                int yy = y + dd[i + 1];
                if (xx < 0 || yy < 0 || xx >= n || yy >= m) continue;
                if (visited[xx][yy]) continue;
                auto newTime = time + 1;
                if (newTime <= moveTime[xx][yy]) {
                    pq.emplace(moveTime[xx][yy] + 1, std::make_pair(xx, yy));
                } else {
                    pq.emplace(newTime, std::make_pair(xx, yy));
                }
            }
        }
        return -1;
    }

    int countBalancedPermutations(const string &num) {
        constexpr int MOD = 1e9 + 7;
        constexpr int MAX_FACT = 81;// 根据最大可能值调整
        vector<long long> fact(MAX_FACT, 0), inv_fact(MAX_FACT, 0);
        auto pow_mod = [&](long long base, long long exp, long long mod) -> long long {
            long long result = 1;
            while (exp > 0) {
                if (exp % 2 == 1) {
                    result = (result * base) % mod;
                }
                base = (base * base) % mod;
                exp /= 2;
            }
            return result;
        };
        auto precompute = [&]() {
            fact[0] = 1;
            for (int i = 1; i < MAX_FACT; ++i) {
                fact[i] = fact[i - 1] * i % MOD;
            }
            inv_fact[MAX_FACT - 1] = pow_mod(fact[MAX_FACT - 1], MOD - 2, MOD);
            for (int i = MAX_FACT - 2; i >= 0; --i) {
                inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD;
            }
        };
        precompute();
        int n = num.size(), sum = 0;
        for (char c: num) sum += c - '0';
        if (sum % 2 == 1) return 0;

        int halfSum = sum / 2, halfLen = n / 2;
        vector<vector<int>> dp(halfSum + 1, vector<int>(halfLen + 1));
        dp[0][0] = 1;

        vector<int> digits(10);
        for (char c: num) {
            int d = c - '0';
            digits[d]++;
            for (int i = halfSum; i >= d; i--)
                for (int j = halfLen; j > 0; j--)
                    dp[i][j] = (dp[i][j] + dp[i - d][j - 1]) % MOD;
        }

        long long res = dp[halfSum][halfLen];
        res = res * fact[halfLen] % MOD * fact[n - halfLen] % MOD;
        for (int i: digits)
            res = res * inv_fact[i] % MOD;
        return res;
    }

    long long minSum(vector<int> &nums1, vector<int> &nums2) {
        long long min_sum1 = 0;
        bool has_zero1 = false;
        long long min_sum2 = 0;
        bool has_zero2 = false;
        for (auto i: nums1) {
            if (i == 0) {
                has_zero1 = true;
                min_sum1 += 1;
            }
            min_sum1 += i;
        }
        for (auto i: nums2) {
            if (i == 0) {
                has_zero2 = true;
                min_sum2 += 1;
            }
            min_sum2 += i;
        }
        if (has_zero1 && has_zero2) {
            return max(min_sum1, min_sum2);
        }
        if (has_zero1) {
            if (min_sum1 > min_sum2) return -1;
            return min_sum2;
        }
        if (has_zero2) {
            if (min_sum2 > min_sum1) return -1;
            return min_sum1;
        }
        if (min_sum1 == min_sum2) return min_sum1;
        return -1;
    }

    bool threeConsecutiveOdds(vector<int> &arr) {
        int consecutive = 0;
        for (auto i: arr) {
            if (i & 1) {
                if (++consecutive >= 3) {
                    return true;
                }
            } else
                consecutive = 0;
        }
        return false;
    }

    int lengthAfterTransformations(const string &s, int t) {
        constexpr char CHAR_BEGIN = 'a';
        constexpr char CHAR_END = 'z';
        constexpr int CHAR_NUM = CHAR_END - CHAR_BEGIN + 1;
        constexpr int MOD = 1e9 + 7;
        vector<long long int> chars(CHAR_NUM, 0);
        for (auto c: s) {
            ++chars[c - CHAR_BEGIN];
        }
        int indexZ = CHAR_END - CHAR_BEGIN;
        for (int k = 0; k < t; ++k) {
            int nextIndexZ = (CHAR_NUM + indexZ - 1) % CHAR_NUM;
            int nextIndexB = (CHAR_NUM + nextIndexZ + 2) % CHAR_NUM;
            chars[nextIndexB] = (chars[nextIndexB] + chars[indexZ]) % MOD;
            indexZ = nextIndexZ;
        }
        return std::accumulate(chars.begin(), chars.end(), 0l) % MOD;
    }

    vector<string> getLongestSubsequence(vector<string> &words, vector<int> &groups) {
        int n = words.size();
        vector<int> prevIndices(n, -1);
        vector<int> dp(n, -1);
        auto f = [&](auto &&f, int index) -> int {
            if (index >= n || index < 0) return 0;
            if (dp[index] != -1) return dp[index];
            int maxPrevIndex = -1;
            int maxPrevLen = 0;
            for (int i = 0; i < index; ++i) {
                if (groups[i] != groups[index]) {
                    int newLen = f(f, i) + 1;
                    if (maxPrevLen < newLen) {
                        maxPrevLen = newLen;
                        maxPrevIndex = i;
                    }
                }
            }
            prevIndices[index] = maxPrevIndex;
            return dp[index] = maxPrevLen;
        };
        for (int i = 0; i < n; ++i) {
            f(f, i);
        }
        vector<string> reverse_res;
        auto index = std::ranges::max_element(dp) - dp.begin();
        while (index != -1) {
            reverse_res.emplace_back(words[index]);
            index = prevIndices[index];
        }
        std::reverse(reverse_res.begin(), reverse_res.end());
        return reverse_res;
    }

    void sortColors(vector<int> &nums) {
        int n = nums.size();
        int right = nums.size() - 1;
        int left = 0;
        int lastTwo = n;
        while (left <= right) {
            while (right >= 0 && nums[right] == 2) {
                --right;
            }
            while (left < n && nums[left] != 2) {
                ++left;
            }
            if (left >= right) {
                break;
            } else {
                std::swap(nums[left], nums[right]);
                ++left;
                --right;
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            if (nums[i] == 2) {
                lastTwo = i;
            } else
                break;
        }
        right = lastTwo - 1;
        left = 0;
        while (left < right) {
            while (right >= 0 && nums[right] == 1) {
                --right;
            }
            while (left < n && nums[left] == 0) {
                ++left;
            }
            if (left >= right) {
                break;
            } else {
                std::swap(nums[left], nums[right]);
                ++left;
                --right;
            }
        }
    }

    int colorTheGrid(int m, int n) {
        constexpr int MOD = 1e9 + 7;
        vector<int> patterns;
        auto generatePattern = [&](auto &&generatePattern, int index, int pattern) -> void {
            if (index == m) {
                patterns.emplace_back(pattern);
                return;
            }
            for (int k = 1; k <= 3; ++k) {
                if (pattern % 10 == k) {
                    continue;
                }
                generatePattern(generatePattern, index + 1, pattern * 10 + k);
            }
        };
        generatePattern(generatePattern, 0, 0);
        int patternsCount = patterns.size();
        unordered_map<int, vector<int>> canConcate;
        for (int i = 0; i < patternsCount; ++i) {
            for (int j = i + 1; j < patternsCount; ++j) {
                bool flag = true;
                auto patternI = patterns[i];
                auto patternJ = patterns[j];
                for (int k = 0; flag && k < m; ++k) {
                    if (patternI % 10 == patternJ % 10)
                        flag = false;
                    patternI /= 10;
                    patternJ /= 10;
                }
                if (flag) {
                    canConcate[i].emplace_back(j);
                    canConcate[j].emplace_back(i);
                }
            }
        }
        vector<int> counts(patternsCount, 1);
        vector<int> temp(patternsCount);
        for (int i = 1; i < n; ++i) {
            for (int k = 0; k < patternsCount; ++k) {
                temp[k] = 0;
                for (int next: canConcate[k]) {
                    temp[k] = (temp[k] + counts[next]) % MOD;
                }
            }
            temp.swap(counts);
        }
        return std::accumulate(counts.begin(), counts.end(), 0ll) % MOD;
    }

    void setZeroes(vector<vector<int>> &matrix) {
        int m = matrix.size();
        int n = matrix.front().size();
        bool firstCol = false;
        bool firstRow = false;
        for (int i = 0; i < m; ++i) {
            if (matrix[i][0] == 0) {
                firstCol = true;
            }
        }
        for (int j = 0; j < n; ++j) {
            if (matrix[0][j] == 0) {
                firstRow = true;
            }
        }

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        for (int i = 1; i < m; ++i) {
            if (matrix[i][0] == 0) {
                for (int j = 0; j < n; ++j) {
                    matrix[i][j] = 0;
                }
            }
        }

        for (int j = 1; j < n; ++j) {
            if (matrix[0][j] == 0) {
                for (int i = 0; i < m; ++i) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (firstCol) {
            for (int i = 0; i < m; ++i) {
                matrix[i][0] = 0;
            }
        }
        if (firstRow) {
            for (int j = 0; j < n; ++j) {
                matrix[0][j] = 0;
            }
        }
    }

    long long maximumValueSum(vector<int> &nums, int k, vector<vector<int>> &edges) {
        long long res = 0;
        res = std::accumulate(nums.begin(), nums.end(), 0ll);
        bool hasZero = false;
        for (auto &i: nums) {
            i = (i ^ k) - i;
            if (i == 0) hasZero = true;
        }
        if (hasZero) {
            for (int i = 0; i < nums.size(); ++i) {
                if (nums[i] >= 0) { res += nums[i]; }
            }
            return res;
        } else {
            int prev = -1;
            int maxNeg = std::numeric_limits<int>::min();
            int minPos = -1;
            for (int i = 0; i < nums.size(); ++i) {
                if (nums[i] < 0) {
                    if (maxNeg < nums[i]) {
                        maxNeg = nums[i];
                    }
                } else if (nums[i] > 0) {
                    if (minPos == -1) minPos = nums[i];
                    else if (minPos > nums[i]) {
                        if (prev == -1) {
                            prev = minPos;
                            minPos = nums[i];
                        } else {
                            res += minPos + prev;
                            prev = -1;
                            minPos = nums[i];
                        }
                    } else {
                        if (prev == -1) {
                            prev = nums[i];
                        } else {
                            res += nums[i] + prev;
                            prev = -1;
                        }
                    }
                }
            }
            if (prev != -1) res += minPos + prev;
            else if (minPos + maxNeg > 0)
                res += minPos + maxNeg;
            return res;
        }
    }

    vector<int> findWordsContaining(vector<string> &words, char x) {
        vector<int> res;
        for (int i = 0; i < words.size(); ++i) {
            if (std::find(words[i].begin(), words[i].end(), x) != words[i].end()) {
                res.emplace_back(i);
            }
        }
        return res;
    }

    int longestPalindrome(vector<string> &words) {
        struct TrieNode {
            TrieNode *next['z' - 'a' + 1] = {nullptr};
            vector<int> indices;
        };

        TrieNode *root = new TrieNode();

        auto contains = [&](const auto &&begin, const auto &&end) {
            auto temp = root;
            for (auto i = begin; i != end; ++i) {
                temp = temp->next[*i - 'a'];
                if (temp == nullptr) {
                    return false;
                }
            }
            return !temp->indices.empty();
        };

        auto add = [&](const string &input, int index) {
            auto temp = root;
            for (int i = 0; i < input.size(); ++i) {
                if (temp->next[input[i] - 'a'] == nullptr)
                    temp->next[input[i] - 'a'] = new TrieNode();
                temp = temp->next[input[i] - 'a'];
            }
            temp->indices.emplace_back(index);
        };

        auto remove = [&](const auto &&begin, const auto &&end) {
            auto temp = root;
            for (auto i = begin; i != end; ++i) {
                if (temp->next[*i - 'a'] == nullptr)
                    temp->next[*i - 'a'] = new TrieNode();
                temp = temp->next[*i - 'a'];
            }
            auto res = temp->indices.back();
            temp->indices.pop_back();
            return res;
        };

        int res = 0;
        auto checkIsPalindrome = [](const auto &word) {
            int j = word.size() - 1;
            int i = 0;
            while (i < j) {
                if (word[i] == word[j]) {
                    ++i;
                    --j;
                } else
                    return false;
            }
            return true;
        };
        int maxPalindrome = 0;
        vector<int> canUseInMid(words.size(), true);
        for (int i = 0; i < words.size(); ++i) {
            auto &s = words[i];
            if (contains(s.rbegin(), s.rend())) {
                res += s.size() * 2;
                canUseInMid[i] = false;
                canUseInMid[remove(s.rbegin(), s.rend())] = false;
            } else {
                if (!checkIsPalindrome(s))
                    canUseInMid[i] = false;
                add(s, i);
            }
        }
        auto maxMid = static_cast<decltype(words.front().size())>(0);
        for (int i = 0; i < words.size(); ++i) {
            if (canUseInMid[i]) {
                maxMid = max(maxMid, words[i].size());
            }
        }
        return res + maxMid;
    }

    int largestPathValue(const string &colors, vector<vector<int>> &edges) {
        auto colorMin = std::ranges::min(colors);
        auto colorMax = std::ranges::max(colors);
        int colorNum = colorMax - colorMin + 1;
        int n = colors.size();
        int m = edges.size();
        vector<vector<int>> dp(n, vector<int>(colorNum, 0));
        vector<int> indegree(n, 0);
        vector<vector<int>> graph(n);
        for (auto i = 0; i < m; ++i) {
            int u = edges[i][0];
            int v = edges[i][1];
            indegree[v]++;
            graph[u].emplace_back(v);
        }
        stack<int> zeroInDegreeNodes;
        int visited = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (indegree[i] == 0) {
                zeroInDegreeNodes.emplace(i);
                res = max(res, dp[i][colors[i] - colorMin] = 1);
                ++visited;
            }
        }
        while (!zeroInDegreeNodes.empty()) {
            auto top = zeroInDegreeNodes.top();
            zeroInDegreeNodes.pop();
            for (auto next: graph[top]) {
                for (auto i = 0; i < colorNum; ++i) {
                    res = max(res, dp[next][i] = max(dp[next][i], dp[top][i] + (i == (colors[next] - colorMin) ? 1 : 0)));
                }
                --indegree[next];
                if (indegree[next] == 0) {
                    ++visited;
                    zeroInDegreeNodes.emplace(next);
                }
            }
        }
        if (visited < n) {
            return -1;
        }
        return res;
    }

    int differenceOfSums(int n, int m) {
        return (((1 + n) * n) >> 1) - (1 + n / m) * m * (n / m);
    }

    vector<int> maxTargetNodes(vector<vector<int>> &edges1, vector<vector<int>> &edges2, int k) {
        int n = edges1.size() + 1;
        int m = edges2.size() + 1;

        auto constructGraph = [](vector<vector<int>> &graph, int nodeNum, vector<vector<int>> &edges) {
            graph.resize(nodeNum);
            for (auto &i: graph) graph.clear();
            for (auto &edge: edges) {
                graph[edge[0]].emplace_back(edge[1]);
                graph[edge[1]].emplace_back(edge[0]);
            }
            return graph;
        };
        vector<vector<int>> graph;

        auto dfs = [&](auto &&dfs, vector<vector<int>> &graph, vector<bool> &visited, int node, int dis) -> int {
            int res = 1;
            if (dis <= 0) return res;
            for (auto next: graph[node]) {
                if (!visited[next]) {
                    visited[next] = true;
                    res += dfs(dfs, graph, visited, next, dis - 1);
                }
            }
            return res;
        };
        vector<bool> visited(m, false);

        int maxTree2 = 0;
        if (k > 0) {
            constructGraph(graph, m, edges2);
            for (int i = 0; i < m; ++i) {
                std::fill(visited.begin(), visited.end(), false);
                visited[i] = true;
                maxTree2 = max(maxTree2, dfs(dfs, graph, visited, i, k - 1));
            }
        }
        constructGraph(graph, n, edges1);
        visited.resize(n);
        vector<int> res(n, 0);
        for (int i = 0; i < n; ++i) {
            std::fill(visited.begin(), visited.end(), false);
            visited[i] = true;
            res[i] = maxTree2 + dfs(dfs, graph, visited, i, k);
        }
        return res;
    }

    int snakesAndLadders(vector<vector<int>> &board) {
        int n = board.size();
        vector<bool> visited(n * n, false);
        auto convertToXY = [&](int index) {
            --index;
            return std::make_pair((n - 1 - index / n), ((index / n) % 2 == 0) ? index % n : n - 1 - index % n);
        };
        vector<int> next;
        vector<int> now;
        int time = 0;
        now.emplace_back(1);
        while (now.size()) {
            while (now.size()) {
                int index = now.back();
                now.pop_back();
                if (visited[index - 1]) continue;
                if (index == n * n) return time;
                visited[index - 1] = true;
                for (int i = index + 1; i <= index + 6 && i <= n * n; ++i) {
                    auto [x, y] = convertToXY(i);
                    if (board[x][y] != -1) {
                        next.emplace_back(board[x][y]);
                    } else
                        next.emplace_back(i);
                }
            }
            next.swap(now);
            ++time;
        }
        return -1;
    }

    long long distributeCandies(int n, int limit) {
        long long res = 0;
        for (int i = 0; i <= min(n, limit); ++i) {
            res += max(0, 1 + min(n - i, limit) - max(0, n - i - limit));
        }
        return res;
    }

    int maxValue(vector<vector<int>> &events, int k) {
        int n = events.size();
        std::sort(events.begin(), events.end());
        auto findNext = [&](auto index) {
            int end = events[index][1];
            int low = index + 1;
            int high = n - 1;
            int next = n;
            while (low <= high) {
                int mid = (low + high) >> 1;
                if (events[mid][0] > end) {
                    next = mid;
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            }
            return next;
        };

        vector<vector<int>> dp(n, vector<int>(k + 1, -1));

        auto dfs = [&](auto &&dfs, int index, int k) {
            if (index >= n || k <= 0) {
                return 0;
            }
            if (dp[index][k] != -1) {
                return dp[index][k];
            }
            int next = findNext(index);
            int maxValue = dfs(dfs, next, k - 1) + events[index][2];
            maxValue = max(maxValue, dfs(dfs, index + 1, k));
            return dp[index][k] = maxValue;
        };
        return dfs(dfs, 0, k);
    }

    int maxFreeTime(int eventTime, int k, vector<int> &startTime, vector<int> &endTime) {
        int n = startTime.size();
        if (k >= n) {
            int res = eventTime;
            for (int i = 0; i < n; ++i) {
                res -= endTime[i] - startTime[i];
            }
            return res;
        }
        auto duration = [&](auto i) {
            return endTime[i] - startTime[i];
        };
        int sum = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int start = 0;
            if (i - k >= 0) {
                start = endTime[i - k];
                sum -= duration(i - k);
            }
            sum += duration(i);
            int end = eventTime;
            if (i + 1 < n) {
                end = startTime[i + 1];
            }
            res = max(res, end - start - sum);
        }
        return res;
    }

    int mostBooked(int n, vector<vector<int>> &meetings) {
        vector<int> count(n, 0);
        std::sort(meetings.begin(), meetings.end());
        long long time = 0;
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> rooms;
        priority_queue<int, vector<int>, greater<>> freeRooms;
        for (int i = 0; i < n; ++i) {
            freeRooms.emplace(i);
        }
        int meetingIndex = 0;
        while (meetingIndex < meetings.size()) {
            if (time < meetings[meetingIndex][0]) {
                time = meetings[meetingIndex][0];
            }
            while (rooms.size() && rooms.top().first <= time) {
                freeRooms.emplace(rooms.top().second);
                rooms.pop();
            }
            if (freeRooms.empty()) {
                time = rooms.top().first;
            } else {
                int chosenRoom = freeRooms.top();
                freeRooms.pop();
                rooms.emplace(time + meetings[meetingIndex][1] - meetings[meetingIndex][0], chosenRoom);
                ++count[chosenRoom];
                ++meetingIndex;
            }
        }
        return std::max_element(count.begin(), count.end()) - count.begin();
    }
};

int main() {
    return 0;
}