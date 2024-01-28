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
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int >>, greater<>>
                minHeap;
        minHeap.emplace(0, 0, 0);
        dist[0][0] = 0;

        int directions[4][2] = {{0,  1},
                                {0,  -1},
                                {1,  0},
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
                      }
        );

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
        vector<tuple<int, int, int >> inputs(n);
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
            if (!node)return;
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
                    stack.emplace(arr[i] * static_cast<long long> (i + 1) % Mod, i);
                } else {
                    stack.emplace(arr[i] * static_cast<long long> (i - stack.top().second ) % Mod, i);
                }
            }
        }
        while (stack.size()) {
            auto [num, index] = stack.top();
            stack.pop();
            updateRes(static_cast<long long> (n - index) % Mod * num);
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
                                  ((j > 0 ? dp[i][j - 1] : 0) + (j < n - 1 ? dp[i][j + 1] : 0)) % M) % M;
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

    int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target)
    {
        int ans = 0;
        int n = matrix.size(), m = matrix[0].size();
        for(int left = 0 ; left < m ; left++)
        {
            vector<int>pre(n, 0);
            for(int right = left ; right < m ; right++)
            {
                for(int i = 0 ; i<n; i++)
                {
                    pre[i] += matrix[i][right];
                }

                for(int i = 0 ; i<n;i++)
                {
                    int sum = 0;
                    for(int j = i ; j<n; j++)
                    {
                        sum += pre[j];
                        if(sum == target)
                        {
                            ans += 1;
                        }
                    }
                }
            }
        }
        return ans;
    }
};

int main() {
    return 0;
}