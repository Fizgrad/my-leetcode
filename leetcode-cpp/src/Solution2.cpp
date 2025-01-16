//
// Created by David Chen on 5/20/23.
//
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <queue>
#include <ratio>
#include <regex>
#include <semaphore>
#include <set>
#include <sstream>
#include <stack>
#include <string.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

class Solution {
public:
    int shortestBridge(vector<vector<int>> &grid) {
        int n = grid.size();
        int xy[5] = {0, 1, 0, -1, 0};

        auto isMargin = [&](int x, int y) -> bool {
            for (int i = 0; i < 4; ++i) {
                int xx = x + xy[i];
                int yy = y + xy[i + 1];
                if (xx >= 0 && yy >= 0 && xx < n && yy < n && grid[xx][yy] == 0) {
                    return true;
                }
            }
            return false;
        };

        vector<pair<int, int>> s;
        vector<vector<bool>> visited(n, vector<bool>(n, false));
        vector<pair<int, int>> margin;

        bool flag = true;
        for (int i = 0; i < n && flag; ++i) {
            for (int j = 0; j < n && flag; ++j) {
                if (grid[i][j] == 1) {
                    s.emplace_back(i, j);
                    visited[i][j] = true;
                    flag = false;
                }
            }
        }
        while (!s.empty()) {
            auto [x, y] = s.back();
            s.pop_back();
            if (isMargin(x, y)) {
                margin.emplace_back(x, y);
            }
            for (int k = 0; k < 4; ++k) {
                int xx = x + xy[k];
                int yy = y + xy[k + 1];
                if (xx >= 0 && yy >= 0 && xx < n && yy < n && grid[xx][yy] == 1 && !visited[xx][yy]) {
                    visited[xx][yy] = true;
                    s.emplace_back(xx, yy);
                }
            }
        }
        vector<pair<int, int>> temp;
        vector<pair<int, int>> next(margin);
        int res = 0;
        while (!next.empty()) {
            temp.clear();
            for (auto [x, y]: next) {
                for (int k = 0; k < 4; ++k) {
                    int xx = x + xy[k];
                    int yy = y + xy[k + 1];
                    if (xx >= 0 && yy >= 0 && xx < n && yy < n) {
                        if (grid[xx][yy] == 1 && !visited[xx][yy]) {
                            return res;
                        }
                        if (grid[xx][yy] == 0 && !visited[xx][yy]) {
                            visited[xx][yy] = true;
                            temp.emplace_back(xx, yy);
                        }
                    }
                }
            }
            next.swap(temp);
            ++res;
        }
        return -1;
    }

    vector<int> topKFrequent(vector<int> &nums, int k) {
        unordered_map<int, int> m;
        for (auto i: nums) {
            ++m[i];
        }
        priority_queue<pair<int, int>> pq;
        for (auto &i: m) {
            pq.emplace(i.second, i.first);
        }
        vector<int> res;
        while (k--) {
            res.push_back(pq.top().second);
            pq.pop();
        }
        return std::move(res);
    }

    long long maxScore(vector<int> &nums1, vector<int> &nums2, int k) {
        int n = nums1.size();
        vector<pair<int, int>> vi2;
        vi2.reserve(n);
        for (int i = 0; i < n; ++i) {
            vi2.emplace_back(nums2[i], i);
        }
        std::sort(vi2.begin(), vi2.end());
        long long int res = 0;
        priority_queue<int> pq;
        long long int sum = 0;
        for (int i = n - 1; i >= 0; --i) {
            sum += nums1[vi2[i].second];
            pq.push(-nums1[vi2[i].second]);
            if (pq.size() > k) {
                sum += pq.top();
                pq.pop();
            }
            if (pq.size() == k)
                res = max(res, sum * vi2[i].first);
        }
        return res;
    }

    double new21Game(int n, int k, int maxPts) {
        // dp和滑动窗口的结合
        if (k == 0 || n >= k + maxPts) {
            return 1.0;
        }
        vector<double> dp(n + 1);
        double currSum = 1.0;
        dp[0] = 1.0;
        for (int i = 1; i <= n; ++i) {
            dp[i] = currSum / maxPts;
            if (i < k) {
                currSum += dp[i];
            }
            if (i - maxPts >= 0) {
                currSum -= dp[i - maxPts];
            }
        }
        double sum = 0.0;
        for (int i = k; i <= n; ++i) {
            sum += dp[i];
        }
        return sum;
    }

    bool isSelfCrossing(vector<int> &distance) {
        if (distance.size() < 4) return false;
        distance.insert(distance.begin(), 0);
        for (int i = 3; i < distance.size(); i++) {
            if (distance[i] >= distance[i - 2] && distance[i - 1] <= distance[i - 3]) return true;
            if (i >= 5) {
                if (distance[i - 1] <= distance[i - 3] && distance[i - 2] >= distance[i - 4] &&
                    distance[i - 5] >= distance[i - 3] - distance[i - 1] &&
                    distance[i] >= distance[i - 2] - distance[i - 4])
                    return true;
            }
        }
        return false;
    }

    int stoneGameII(vector<int> &piles) {
        int n = piles.size();
        vector<int> sum_piles(n);
        sum_piles[n - 1] = piles[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            sum_piles[i] += sum_piles[i + 1] + piles[i];
        }
        vector<vector<int>> dp(101, vector<int>(101, -1));
        auto f = [&](auto &&f, int M, int index) -> int {
            if (index >= n) {
                return 0;
            }
            if (index + 2 * M >= dp.size()) {
                return sum_piles[index];
            }
            if (dp[index][M] != -1) {
                return dp[index][M];
            }
            int res = 0;
            for (int i = 1; i <= 2 * M; ++i) {
                res = max(res, sum_piles[index] - f(f, max(i, M), index + i));
            }
            return dp[index][M] = res;
        };
        return f(f, 1, 0);
    }

    int stoneGameV(vector<int> &stoneValue) {
        int n = stoneValue.size();
        vector<int> prefix_sum(n);
        prefix_sum[0] = stoneValue[0];
        for (int i = 1; i < n; ++i) {
            prefix_sum[i] = prefix_sum[i - 1] + stoneValue[i];
        }
        int dp[501][501] = {0};
        auto f = [&](auto &&f, int left, int right) -> int {
            if (left >= right) {
                return 0;
            }
            if (left == right - 1) {
                return min(stoneValue[left], stoneValue[right]);
            }
            if (dp[left][right]) {
                return dp[left][right];
            }
            int interval_sum = prefix_sum[right];
            if (left) {
                interval_sum -= prefix_sum[left - 1];
            }
            int res = 0;
            for (int i = left + 1; i <= right; ++i) {
                int right_sum = prefix_sum[right] - prefix_sum[i - 1];
                int left_sum = interval_sum - right_sum;
                if (right_sum > left_sum) {
                    res = max(res, left_sum + f(f, left, i - 1));
                } else if (right_sum < left_sum) {
                    res = max(res, right_sum + f(f, i, right));
                } else {
                    res = max(res, max(right_sum + f(f, left, i - 1), left_sum + f(f, i, right)));
                }
            }
            return dp[left][right] = res;
        };
        return f(f, 0, n - 1);
    }

    int maxPerformance(int n, vector<int> &speed, vector<int> &efficiency, int k) {
        vector<pair<int, int>> efficiency_v_i(n);
        for (int i = 0; i < n; ++i) {
            efficiency_v_i[i] = {efficiency[i], i};
        }
        std::sort(efficiency_v_i.begin(), efficiency_v_i.end());
        long long int res = 0;
        priority_queue<int> pq;
        long long int cur_sum = 0;
        for (int i = n - 1; i >= 0; --i) {
            cur_sum += speed[efficiency_v_i[i].second];
            pq.push(-speed[efficiency_v_i[i].second]);
            if (pq.size() > k) {
                cur_sum += pq.top();
                pq.pop();
            }
            res = max(res, efficiency_v_i[i].first * cur_sum);
        }
        return res % 1000000007;
    }

    string stoneGameIII(vector<int> &stoneValue) {
        int n = stoneValue.size();
        vector<int> dp(3, 0);
        int res = 0;
        for (int i = n - 1; i >= 0; --i) {
            int cur_sum = stoneValue[i];
            res = cur_sum - dp[0];
            for (int k = 2; k <= 3 && i + k <= n; ++k) {
                cur_sum += stoneValue[i + k - 1];
                res = max(res, cur_sum - dp[k - 1]);
            }
            dp[2] = dp[1];
            dp[1] = dp[0];
            dp[0] = res;
        }
        return res > 0 ? "Alice" : (res == 0 ? "Tie" : "Bob");
    }

    int stoneGameVIII(vector<int> &stones) {
        int n = stones.size();
        for (int i = 1; i < n; ++i) {
            stones[i] += stones[i - 1];
        }
        if (n == 2) {
            return stones.back();
        }
        int last = stones[n - 1];
        int max_temp = stones[n - 1];
        for (int i = n - 2; i >= 1; --i) {
            int now = max(max_temp, stones[i] - last);
            max_temp = max(max_temp, now);
            last = now;
        }
        return last;
    }

    bool winnerSquareGame(int n) {
        vector<bool> dp(n + 1, -1);
        for (int i = 1; i <= n; ++i) {
            int root = ::sqrt(i);
            if (root * root == i) {
                dp[i] = true;
            } else {
                bool res = false;
                for (int j = 1; i - j * j >= 1; ++j) {
                    if (!dp[i - j * j]) {
                        res = true;
                        break;
                    }
                }
                dp[i] = res;
            }
        }
        return dp[n];
    }

    int minCost(int n, vector<int> &cuts) {
        int m = cuts.size();
        cuts.push_back(0);
        cuts.push_back(n);
        std::sort(cuts.begin(), cuts.end());
        int res = 0;
        int N = m + 1;
        int dp[101][101];
        ::memset(dp, 0, sizeof(dp));
        for (int k = 1; k < N; ++k) {
            for (int i = 0; i < N && i + k < N; ++i) {
                int len = cuts[i + k + 1] - cuts[i];
                dp[i][i + k] = len + dp[i + 1][i + k];
                for (int j = i + 1; j + 1 <= i + k; ++j) {
                    dp[i][i + k] = min(dp[i][i + k], len + dp[i][j] + dp[j + 1][i + k]);
                }
            }
        }
        return dp[0][m];
    }

    int shortestPathBinaryMatrix(vector<vector<int>> &grid) {
        if (grid[0][0] != 0) {
            return -1;
        }
        int n = grid.size();
        vector<vector<bool>> visited(n, vector<bool>(n, false));
        visited[0][0] = true;
        vector<pair<int, int>> next;
        vector<pair<int, int>> temp;
        int dx[8] = {0, 0, 1, 1, 1, -1, -1, -1};
        int dy[8] = {1, -1, 0, 1, -1, 0, 1, -1};
        next.emplace_back(0, 0);
        int res = 0;
        while (!next.empty()) {
            ++res;
            temp.clear();
            while (!next.empty()) {
                auto [x, y] = next.back();
                if (x == n - 1 && y == n - 1) {
                    return res;
                }
                next.pop_back();
                for (int i = 0; i < 8; ++i) {
                    int xx = x + dx[i];
                    int yy = y + dy[i];
                    if (xx >= 0 && yy >= 0 && xx < n && yy < n && !visited[xx][yy] && grid[xx][yy] == 0) {
                        if (xx == n - 1 && yy == n - 1) {
                            return res + 1;
                        }
                        visited[xx][yy] = true;
                        temp.emplace_back(xx, yy);
                    }
                }
            }
            temp.swap(next);
        }
        return -1;
    }

    int maximumDetonation(vector<vector<int>> &bombs) {
        int n = bombs.size();
        vector<vector<int>> graph(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                if (static_cast<double>(bombs[i][0] - bombs[j][0]) * (bombs[i][0] - bombs[j][0]) +
                            static_cast<double>(bombs[i][1] - bombs[j][1]) * (bombs[i][1] - bombs[j][1]) -
                            static_cast<double>(bombs[i][2]) * bombs[i][2] <=
                    0) {
                    graph[i].push_back(j);
                }
            }
        }
        int res = 1;
        for (int i = 0; i < n; ++i) {
            vector<bool> visited(n, false);
            vector<int> stack;
            stack.push_back(i);
            visited[i] = true;
            int temp = 0;
            while (!stack.empty()) {
                ++temp;
                int now = stack.back();
                stack.pop_back();
                for (auto next: graph[now]) {
                    if (!visited[next]) {
                        stack.push_back(next);
                        visited[next] = true;
                    }
                }
            }
            res = max(res, temp);
        }
        return res;
    }

    int maxPathSum(TreeNode *root) {
        int res = root->val;
        auto f = [&](auto &&f, TreeNode *node) -> int {
            if (node == nullptr) {
                return 0;
            }
            int left = f(f, node->left);
            int right = f(f, node->right);
            res = max(res, node->val + max(max(left + right, max(left, right)), 0));
            return max(max(left, right), 0) + node->val;
        };
        f(f, root);
        return res;
    }

    int findCircleNum(vector<vector<int>> &isConnected) {
        int n = isConnected.size();
        vector<bool> isVisited(n, false);
        int res = 0;
        auto dfs = [&](auto dfs, int now) -> void {
            for (int j = 0; j < n; ++j) {
                if (isVisited[j]) {
                    continue;
                } else if (isConnected[now][j]) {
                    isVisited[j] = true;
                    dfs(dfs, j);
                }
            }
        };
        for (int i = 0; i < n; ++i) {
            if (isVisited[i]) {
                continue;
            } else {
                ++res;
                isVisited[i] = true;
                dfs(dfs, i);
            }
        }
        return res;
    }

    bool checkStraightLine(vector<vector<int>> &coordinates) {
        int n = coordinates.size();
        if (n <= 2) {
            return true;
        }
        if (coordinates[0][0] == coordinates[1][0]) {
            for (int i = 2; i < n; ++i) {
                if (coordinates[i][0] == coordinates[1][0]) {
                    continue;
                } else {
                    return false;
                }
            }
            return true;
        }
        double k = static_cast<double>(coordinates[0][1] - coordinates[1][1]) / (coordinates[0][0] - coordinates[1][0]);
        for (int i = 2; i < n; ++i) {
            double temp = static_cast<double>(coordinates[i][1] - coordinates[1][1]) /
                          (coordinates[i][0] - coordinates[1][0]);
            if (k == temp) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }

    bool canMakeArithmeticProgression(vector<int> &arr) {
        std::sort(arr.begin(), arr.end());
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[i] - arr[i - 1] != arr[1] - arr[0]) {
                return false;
            }
        }
        return true;
    }

    int minFlips(int a, int b, int c) {
        return __builtin_popcount(c - ((a | b) - ((a | b | c) ^ c))) +
               __builtin_popcount(a & ((a | b | c) ^ c)) +
               __builtin_popcount(b & ((a | b | c) ^ c));
    }

    int countNegatives(vector<vector<int>> &grid) {
        int num = 0;
        int r = 0;
        int c = 0;
        int n = grid.size();
        int m = grid.begin()->size();
        while (c < m && grid[0][c] >= 0) {
            ++c;
        }
        num += c;
        for (r = 1; r < n; ++r) {
            while (c > 0 && grid[r][c - 1] < 0) {
                --c;
            }
            num += c;
        }
        return m * n - num;
    }

    char nextGreatestLetter(vector<char> &letters, char target) {
        int low = 0;
        int high = letters.size() - 1;
        if (letters[0] > target)
            return letters[0];
        else if (target >= letters[letters.size() - 1])
            return letters[0];
        char ans;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (letters[mid] > target) {
                ans = letters[mid];
                high = mid - 1;
            } else
                low = mid + 1;
        }
        return ans;
    }

    vector<string> summaryRanges(vector<int> &nums) {
        vector<string> res;
        bool flag = false;
        int begin = 0;
        int prev = 0;
        for (auto i: nums) {
            if (!flag) {
                flag = true;
                prev = begin = i;
            } else if (i == prev + 1) {
                prev = i;
            } else if (i > prev + 1) {
                if (prev != begin) res.push_back(to_string(begin) + "->" + to_string(prev));
                else
                    res.push_back(to_string(begin));
                prev = begin = i;
            }
        }
        if (flag) {
            flag = false;
            if (prev != begin) res.push_back(to_string(begin) + "->" + to_string(prev));
            else
                res.push_back(to_string(begin));
        }
        return std::move(res);
    }

    int maxValue(int n, long long int index, long long int maxSum) {
        maxSum -= n;
        int l = 0;
        int r = maxSum;
        while (l != r) {
            long long int m = (l + r + 1) / 2;
            long long int left = min(index, m);
            long long int right = min(m, n - index - 1);
            if (maxSum >= ((m * 2 - 1 - left) * left + (m * 2 - 1 - right) * right) / 2 + m)
                l = m;
            else
                r = m - 1;
        }
        return r + 1;
    }

    int equalPairs(vector<vector<int>> &grid) {
        map<vector<int>, int> r;
        map<vector<int>, int> c;
        for (int i = 0; i < grid.size(); i++) {
            if (r.count(grid[i])) {
                r[grid[i]]++;
            } else {
                r[grid[i]] = 1;
            }
            vector<int> t;
            for (int j = 0; j < grid.size(); j++) {
                t.push_back(grid[j][i]);
            }
            if (c.count(t)) {
                c[t]++;
            } else {
                c[t] = 1;
            }
        }
        int ans = 0;
        for (auto ele: r) {
            if (c.count(ele.first)) {
                ans += ele.second * c[ele.first];
            }
        }
        return ans;
    }

    int getMinimumDifference(TreeNode *root) {
        int prev = -1;
        int res = INT32_MAX;
        auto dfs = [&](auto &&dfs, TreeNode *node) -> void {
            if (node) {
                if (node->left) {
                    dfs(dfs, node->left);
                }
                if (prev != -1) {
                    res = min(res, node->val - prev);
                }
                prev = node->val;
                if (node->right) {
                    dfs(dfs, node->right);
                }
            }
        };
        dfs(dfs, root);
        return res;
    }

    int maxLevelSum(TreeNode *root) {
        int res = 1;
        int level = 0;
        int max_sum = root->val;
        vector<TreeNode *> next;
        vector<TreeNode *> temp;
        next.push_back(root);
        while (!next.empty()) {
            ++level;
            int sum = 0;
            while (!next.empty()) {
                auto i = next.back();
                next.pop_back();
                if (i->right) {
                    temp.push_back(i->right);
                }
                if (i->left) {
                    temp.push_back(i->left);
                }
                sum += i->val;
            }
            next.swap(temp);
            temp.clear();
            if (sum > max_sum) {
                max_sum = sum;
                res = level;
            }
        }
        return res;
    }

    vector<bool> canMakePaliQueries(const string &s, vector<vector<int>> &queries) {
        vector<bool> res(queries.size(), false);
        int n = s.size();
        vector<int> prefix(n, 0);
        prefix[0] = 1 << (s[0] - 'a');
        for (int i = 1; i < n; ++i) {
            prefix[i] = prefix[i - 1] ^ (1 << (s[i] - 'a'));
        }
        for (int i = 0; i < queries.size(); ++i) {
            int temp = 0;
            int start = queries[i][0];
            int end = queries[i][1];
            if (start > 0) {
                temp = __builtin_popcount(prefix[end] ^ prefix[start - 1]);
            } else {
                temp = __builtin_popcount(prefix[end]);
            }
            if (queries[i][2] >= temp / 2) {
                res[i] = true;
            }
        }
        return std::move(res);
    }

    TreeNode *replaceValueInTree(TreeNode *root) {
        vector<TreeNode *> temp;
        vector<TreeNode *> next;
        temp.push_back(root);
        root->val = 0;
        while (!temp.empty()) {
            int sum = 0;
            for (auto &i: temp) {
                if (i->left) {
                    sum += i->left->val;
                }
                if (i->right) {
                    sum += i->right->val;
                }
            }
            while (!temp.empty()) {
                auto i = temp.back();
                temp.pop_back();
                int sum_par = 0;
                if (i->left) {
                    sum_par += i->left->val;
                }
                if (i->right) {
                    sum_par += i->right->val;
                }
                if (i->left) {
                    i->left->val = sum - sum_par;
                    next.push_back(i->left);
                }
                if (i->right) {
                    i->right->val = sum - sum_par;
                    next.push_back(i->right);
                }
            }
            temp.swap(next);
            next.clear();
        }
        return root;
    }

    string countOfAtoms(const string &formula) {
        map<string, int> res;
        vector<string> token;
        string temp;
        int type = 0;// 0: name  1: digits
        for (int i = 0; i < formula.size(); ++i) {
            if (isupper(formula[i])) {
                if (!temp.empty())
                    token.push_back(temp);
                temp.clear();
                type = 0;
                temp.push_back(formula[i]);
            }
            if (isdigit(formula[i])) {
                if (type == 0) {
                    if (!temp.empty())
                        token.push_back(temp);
                    temp.clear();
                    type = 1;
                    temp.push_back(formula[i]);
                } else {
                    temp.push_back(formula[i]);
                }
            }
            if (islower(formula[i])) {
                temp.push_back(formula[i]);
            }
            if (formula[i] == '(' || formula[i] == ')') {
                if (!temp.empty())
                    token.push_back(temp);
                temp.clear();
                token.emplace_back(1, formula[i]);
            }
        }
        if (!temp.empty())
            token.push_back(temp);
        temp.clear();
        vector<int> nums = {1};
        int num = 1;
        for (auto i = token.rbegin(); i != token.rend(); ++i) {
            if (isdigit(*i->begin())) {
                num = stoi(*i);
            }
            if (isalpha(*i->begin())) {
                res[*i] += num * nums.back();
                num = 1;
            }
            if (')' == *i->begin()) {
                nums.push_back(num * nums.back());
                num = 1;
            }

            if ('(' == *i->begin()) {
                nums.pop_back();
            }
        }
        string string_res;
        for (auto &i: res) {
            string_res += i.first;
            if (i.second != 1) {
                string_res += to_string(i.second);
            }
        }
        return std::move(string_res);
    }

    int numOfWays(vector<int> &nums) {
        constexpr int MOD = 1e9 + 7;
        // 快速幂求 (x^y) % MOD
        auto fast_pow = [&](int x, int y) -> int {
            int res = 1;
            while (y > 0) {
                if (y & 1) {
                    res = (1LL * res * x) % MOD;
                }
                x = (1LL * x * x) % MOD;
                y >>= 1;
            }
            return res;
        };

        // 求 a 在模 MOD 下的乘法逆元，即 (a^-1) % MOD
        auto mod_inverse = [&](int a) -> int {
            return fast_pow(a, MOD - 2);// 根据费马小定理
        };

        // 计算 C(n, k) % MOD
        auto nCk_mod = [&](int n, int k) {
            int numerator = 1;// 分子，保存 n! % MOD
            for (int i = 1; i <= n; ++i) {
                numerator = (1LL * numerator * i) % MOD;
            }

            int denominator = 1;// 分母，保存 (k! * (n-k)!) % MOD
            for (int i = 1; i <= k; ++i) {
                denominator = (1LL * denominator * i) % MOD;
            }
            for (int i = 1; i <= n - k; ++i) {
                denominator = (1LL * denominator * i) % MOD;
            }

            // 通过乘法逆元将除法转换为乘法
            return (1LL * numerator * mod_inverse(denominator)) % MOD;
        };

        auto dfs = [&](auto &&dfs, vector<int> &arr) -> int {
            if (arr.size() <= 1) {
                return 1;
            }
            int mid = *arr.begin();
            vector<int> right;
            vector<int> left;
            for (auto i: arr) {
                if (i > mid) {
                    right.push_back(i);
                }
                if (i < mid) {
                    left.push_back(i);
                }
            }
            int res = nCk_mod(right.size() + left.size(), left.size());
            res = 1LL * res * (dfs(dfs, left) % MOD) % MOD;
            res = 1LL * res * (dfs(dfs, right) % MOD) % MOD;
            return res;
        };
        return dfs(dfs, nums) - 1;
    }

    int countPaths(vector<vector<int>> &grid) {
        constexpr int mod = 1e9 + 7;
        int res = 0;
        int n = grid.size();
        int m = grid.front().size();
        vector<vector<int>> dp(n, vector<int>(m, -1));
        constexpr int dx[4] = {0, 0, 1, -1};
        constexpr int dy[4] = {1, -1, 0, 0};
        auto f = [&](auto &&f, int x, int y) {
            if (x < 0 || y < 0 || x >= n || y >= m) {
                return 0;
            } else if (dp[x][y] != -1) {
                return dp[x][y];
            } else {
                dp[x][y] = 1;
                for (int k = 0; k < 4; ++k) {
                    int xx = x + dx[k];
                    int yy = y + dy[k];
                    if (xx < 0 || yy < 0 || xx >= n || yy >= m)
                        continue;
                    if (grid[x][y] < grid[xx][yy]) {
                        dp[x][y] = (dp[x][y] + f(f, xx, yy)) % mod;
                    }
                }
                return dp[x][y] % mod;
            }
        };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                res = (res + f(f, i, j)) % mod;
            }
        }
        return res;
    }

    int largestAltitude(vector<int> &gain) {
        int res = 0;
        int temp = 0;
        for (auto i: gain) {
            temp += i;
            res = max(res, temp);
        }
        return res;
    }

    vector<int> getAverages(vector<int> &nums, int k) {
        int n = nums.size();
        if (2 * k >= n) {
            return std::move(vector<int>(n, -1));
        }
        vector<int> res(n, -1);
        long long int k_sum = 0;
        for (int i = 0; i <= 2 * k; ++i) {
            k_sum += nums[i];
        }
        res[k] = k_sum / (2 * k + 1);
        for (int i = k + 1; i + k < n; ++i) {
            k_sum += (-nums[i - k - 1] + nums[i + k]);
            res[i] = k_sum / (2 * k + 1);
        }
        return std::move(res);
    }

    long long minCost(vector<int> &nums, vector<int> &cost) {
        auto calc = [&](long long median) -> long long {
            long long ans = 0;
            for (int i = 0; i < nums.size(); i++)
                ans += abs(1ll * nums[i] - median) * (1ll * cost[i]);
            return ans;
        };
        long long tot = 0;
        long long sum = 0;
        int n = nums.size();
        vector<pair<int, int>> vec;
        for (int i = 0; i < nums.size(); i++)
            vec.emplace_back(nums[i], cost[i]);
        sort(vec.begin(), vec.end());
        for (int i = 0; i < n; i++)
            sum += vec[i].second;
        long long median;
        int i = 0;
        while (tot < (sum + 1) / 2 && i < n) {
            tot += 1ll * vec[i].second;
            median = vec[i].first;
            i++;
        }
        return calc(median);
    }

    int maxProfit(vector<int> &prices, int fee) {
        int hold = -prices[0];
        int not_hold = 0;
        for (auto i: prices) {
            not_hold = max(not_hold, hold + i - fee);
            hold = max(hold, not_hold - i);
        }
        return max(hold, not_hold);
    }

    int tallestBillboard(vector<int> &rods) {
        int sum = accumulate(rods.begin(), rods.end(), 0);
        vector<int> dp(sum / 2 + 1, -1);
        vector<int> temp(sum / 2 + 1, -1);
        temp[0] = dp[0] = 0;
        for (auto i: rods) {
            for (int j = 0; j <= sum / 2; ++j) {
                if (dp[j] == -1) {
                    continue;
                }
                if (i + j <= sum / 2)
                    temp[j + i] = max(temp[j + i], dp[j]);
                if (i > j) {
                    //在这里使用temp的原因是，在之前可能会存在其他情况更新了temp[i-j]或者temp[j-i],
                    //因为我们不能保证i-j, j-i的值在之前我们没修改，比如 i == 2 j == 4的时候
                    //temp[4-2]在之前 temp[2-0]的时候也修改了，我们不能确定最终是谁最大，如果用的
                    //temp[i - j] = max(dp[i - j], dp[j] + j);
                    //temp[j - i] = max(dp[j - i], dp[j] + i);
                    //我们会丢掉之前temp[2-0]的信息
                    if (i - j <= sum / 2)
                        temp[i - j] = max(temp[i - j] /* ！！！ */, dp[j] + j);
                } else {
                    temp[j - i] = max(temp[j - i] /* ！！！ */, dp[j] + i);
                }
            }
            dp = temp;
        }
        return dp[0];
    }

    int countRoutes(vector<int> &locations, int start, int finish, int fuel) {
        constexpr int mod = 1e9 + 7;
        int n = locations.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(fuel + 1, -1)));
        auto f = [&](auto &&f, int from, int to, int remaining_fuel) {
            if (dp[from][to][remaining_fuel] != -1) {
                return dp[from][to][remaining_fuel];
            }
            long long int ans = (from == to);
            for (int i = 0; i < n; ++i) {
                if (i == from) {
                    continue;
                }
                int dis = ::abs(locations[i] - locations[from]);
                if (dis > remaining_fuel) {
                    continue;
                }
                ans = (ans + f(f, i, to, remaining_fuel - dis)) % mod;
            }
            return dp[from][to][remaining_fuel] = ans % mod;
        };
        return f(f, start, finish, fuel);
    }

    long long totalCost(vector<int> &costs, int k, int candidates) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        vector<pair<int, int>> arr;
        int n = costs.size();
        for (int i = 0; i < n; ++i) {
            arr.emplace_back(costs[i], i);
        }
        if (n >= 2 * candidates)
            for (int i = 0; i < candidates; ++i) {
                pq.push(arr[i]);
                pq.push(arr[n - i - 1]);
            }
        else
            for (auto &i: arr) pq.push(i);
        long long res = 0;
        int left = candidates;
        int right = n - candidates - 1;
        while (k--) {
            auto [val, idx] = pq.top();
            pq.pop();
            res += val;
            if (left <= right) {
                if (idx < left) {
                    pq.push(arr[left++]);
                } else if (idx > right) {
                    pq.push(arr[right--]);
                }
            }
        }
        return res;
    }

    vector<vector<int>> kSmallestPairs(vector<int> &nums1, vector<int> &nums2, int k) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        for (auto i: nums1) {
            pq.push({i + nums2[0], 0});
            if (pq.size() >= k) {
                break;
            }
        }
        vector<vector<int>> res;
        while (k-- && !pq.empty()) {
            auto [sum, j] = pq.top();
            pq.pop();
            res.push_back({sum - nums2[j], nums2[j]});
            if (j + 1 < nums2.size())
                pq.push({sum - nums2[j] + nums2[j + 1], j + 1});
        }
        return std::move(res);
    }

    double maxProbability(int n, vector<vector<int>> &edges, vector<double> &succProb, int start, int end) {
        vector<vector<pair<int, double>>> graph(n);
        for (auto &i: succProb) {
            i = -::log(i) / ::log(2);
        }
        for (int i = 0; i < edges.size(); ++i) {
            graph[edges[i][0]].emplace_back(edges[i][1], succProb[i]);
            graph[edges[i][1]].emplace_back(edges[i][0], succProb[i]);
        }
        vector<double> prob(n, -1.0);
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
        pq.emplace(0, start);
        while (!pq.empty()) {
            auto [pb, now] = pq.top();
            pq.pop();
            prob[now] = pb;
            if (now == end) {
                return ::pow(2, -pb);
            }
            for (auto &k: graph[now]) {
                if (prob[k.first] < 0.0) {
                    pq.emplace(pb + k.second, k.first);
                }
            }
        }
        return 0.0;
    }

    int shortestPathAllKeys(vector<string> &grid) {
        int m = grid.size();
        int n = grid.front().size();
        constexpr int dx[4] = {0, 0, -1, 1};
        constexpr int dy[4] = {1, -1, 0, 0};
        int start_x = 0;
        int start_y = 0;
        int num_keys = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (islower(grid[i][j])) {
                    num_keys = max(num_keys, grid[i][j] - 'a' + 1);
                }
                if (grid[i][j] == '@') {
                    start_x = i;
                    start_y = j;
                }
            }
        }
        int keys_all = (1 << num_keys) - 1;
        vector<vector<vector<int>>> dp(m, vector<vector<int>>(n, vector<int>(1 << num_keys, -1)));
        queue<tuple<int, int, int, int>> s;
        auto bfs = [&](int keys, int x, int y, int path) {
            for (int k = 0; k < 4; ++k) {
                int xx = x + dx[k];
                int yy = y + dy[k];
                if (xx >= 0 && xx < m && yy >= 0 && yy < n) {
                    int keys_ = keys;
                    char c = grid[xx][yy];
                    if (c == '#') { continue; }
                    if (islower(c)) { keys_ |= (1 << (c - 'a')); }
                    if (isupper(c)) {
                        if ((keys_ & (1 << (c - 'A'))) == 0) {
                            continue;
                        }
                    }
                    if (dp[xx][yy][keys_] == -1) {
                        dp[xx][yy][keys_] = path;
                        s.emplace(keys_, xx, yy, path + 1);
                    }
                }
            }
        };
        s.emplace(0, start_x, start_y, 0);
        while (!s.empty()) {
            auto [keys, x, y, len] = s.front();
            s.pop();
            if (keys == keys_all)
                return len;
            bfs(keys, x, y, len);
        }
        return -1;
    }

    int latestDayToCross(int row, int col, vector<vector<int>> &cells) {

        constexpr int dx[8] = {0, 0, 1, 1, 1, -1, -1, -1};
        constexpr int dy[8] = {1, -1, 1, 0, -1, 1, 0, -1};

        int low = col - 1;
        int high = cells.size();
        //        vector<unordered_set<int>> graph(col, unordered_set < int > ());
        //        int prev_day = -1;
        auto f = [&](int mid) {
            //            if (prev_day == -1) {
            //                for (int i = 0; i < mid; ++i) {
            //                    graph[cells[i][1] - 1].insert(cells[i][0] - 1);
            //                }
            //                prev_day = mid;
            //            } else if (prev_day < mid) {
            //                for (int i = prev_day; i < mid; ++i) {
            //                    graph[cells[i][1] - 1].insert(cells[i][0] - 1);
            //                }
            //                prev_day = mid;
            //            } else {
            //                for (int i = mid; i < prev_day; ++i) {
            //                    graph[cells[i][1] - 1].erase(cells[i][0] - 1);
            //                }
            //                prev_day = mid;
            //            }
            vector<unordered_set<int>> graph(col, unordered_set<int>());
            for (int i = 0; i < mid; ++i) {
                graph[cells[i][1] - 1].insert(cells[i][0] - 1);
            }
            vector<pair<int, int>> s;
            for (auto i: graph[0]) {
                s.emplace_back(i, 0);
            }
            while (!s.empty()) {
                auto [x, y] = s.back();
                s.pop_back();
                for (int k = 0; k < 8; ++k) {
                    int xx = x + dx[k];
                    int yy = y + dy[k];
                    if (yy > 0 && yy < col && xx >= 0 && xx < row) {
                        if (graph[yy].count(xx)) {
                            if (yy == col - 1) {
                                return false;
                            }
                            s.emplace_back(xx, yy);
                            graph[yy].erase(xx);
                        }
                    }
                }
            }
            return true;
        };

        while (low >= 0 && low < high) {
            int mid = (low + high + 1) / 2;
            if (f(mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    int distributeCookies(vector<int> &cookies, int k) {
        vector<int> children(k, 0);
        int n = cookies.size();
        int res = INT32_MAX;
        auto dfs = [&](auto &&dfs, int index) {
            if (index == n) {
                res = min(res, *std::max_element(children.begin(), children.end()));
                return;
            } else {
                int cookie = cookies[index];
                for (int i = 0; i < k; ++i) {
                    children[i] += cookie;
                    if (children[i] < res)
                        dfs(dfs, index + 1);
                    children[i] -= cookie;
                }
            }
        };
        dfs(dfs, 0);
        return res;
    }

    bool buddyStrings(const string &s, const string &goal) {
        int n = s.size();
        char c = 0;
        char cc = 0;
        bool count[26] = {0};
        bool flag = false;
        if (s.size() != goal.size()) {
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (s[i] == goal[i]) {
                if (!flag) {
                    count[s[i] - 'a'] = !count[s[i] - 'a'];
                    if (!count[s[i] - 'a']) {
                        flag = true;
                    }
                }
                continue;
            } else {
                if (!c) {
                    c = s[i];
                    cc = goal[i];
                } else {
                    if (c != goal[i] || cc != s[i]) {
                        return false;
                    }
                    c = cc = 'A';
                }
            }
        }
        return c == 'A' || (c == 0 && flag);
    }

    int maximumRequests(int n, vector<vector<int>> &requests) {
        int req_num = requests.size();
        vector<int> degree(n, 0);
        int bit = 0;
        int res = 0;
        auto dfs = [&](auto &&dfs, int index) {
            if (index >= req_num) {
                if (all_of(begin(degree), end(degree), [](int item) {
                        return item == 0;
                    }))
                    res = max(res, __builtin_popcount(bit));
                return;
            } else {
                if (res >= __builtin_popcount(bit) + req_num - index)
                    return;
                int from = requests[index][0];
                int to = requests[index][1];
                bit |= (1 << index);
                --degree[from];
                ++degree[to];
                dfs(dfs, index + 1);
                ++degree[from];
                --degree[to];
                bit -= (1 << index);
                dfs(dfs, index + 1);
            }
        };
        dfs(dfs, 0);
        return res;
    }

    int singleNumber(vector<int> &nums) {
        int ones = 0;
        int twos = 0;
        int threes = 0;
        for (auto num: nums) {
            //twos keep the bits which appear twice
            twos |= ones & num;
            // ones keep the bits which appear only once
            ones ^= num;
            // threes represent whether one bit has appeared three times
            threes = ones & twos;
            //if one bit has appeared three times, we clear the corresponding bits in both ones and twos
            ones &= ~threes;
            twos &= ~threes;
        }
        return ones;
    }

    vector<int> singleNumberII(vector<int> &nums) {
        long long int xor_all = 0;
        for (auto i: nums) {
            xor_all ^= i;
        }
        int ans1 = 0, ans2 = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (xor_all & (-xor_all) & nums[i])
                ans1 ^= nums[i];
            else
                ans2 ^= nums[i];
        }
        vector<int> res = {ans1, ans2};
        return std::move(res);
    }

    int longestSubarray(vector<int> &nums) {
        int now = 0;
        int prev = 0;
        int res = 0;
        int zeros = 0;
        bool flag = false;
        for (auto i: nums) {
            if (i == 1) {
                zeros = 0;
                ++now;
                res = max(res, now + prev);
            } else {
                flag = true;
                ++zeros;
                prev = now;
                now = 0;
                if (zeros > 1) {
                    prev = 0;
                }
            }
        }
        return res - !flag;
    }

    string removeDuplicateLetters(const string &s) {
        int n = s.size();
        int last[26] = {0};
        for (int i = 0; i < n; ++i) {
            last[s[i] - 'a'] = i;
        }
        int seen[26] = {0};
        string stack = "";
        for (int i = 0; i < n; ++i) {
            while (stack.size() and stack.back() > s[i] and last[stack.back() - 'a'] > i and seen[s[i] - 'a'] == 0) {
                seen[stack.back() - 'a'] = 0;
                stack.pop_back();
            }
            if (seen[s[i] - 'a'] == 0)
                stack.push_back(s[i]), seen[s[i] - 'a'] = 1;
        }
        return std::move(stack);
    }

    int minSubArrayLen(int target, vector<int> &nums) {
        int sum = 0;
        int n = nums.size();
        int left = 0, right = 0;
        int res = INT32_MAX;
        while (true) {
            if (sum >= target) {
                res = min(res, right - left);
                sum -= nums[left++];
            } else {
                if (right < n)
                    sum += nums[right++];
                else
                    break;
            }
        }
        return res == INT32_MAX ? 0 : res;
    }

    int maxConsecutiveAnswers(const string &answerKey, int k) {
        int n = answerKey.size();
        int res = 0;
        char targets[2] = {'T', 'F'};
        for (auto target: targets) {
            int remains_k = k;
            int left = 0;
            int right = 0;
            while (true) {
                if (remains_k >= 0) {
                    res = max(res, right - left);
                }
                if (right == n) {
                    break;
                }
                if (remains_k >= 0 && right < n) {
                    if (answerKey[right++] != target) {
                        --remains_k;
                    }
                }
                if (remains_k < 0) {
                    if (answerKey[left++] != target) {
                        ++remains_k;
                    }
                }
            }
        }
        return res;
    }

    long long putMarbles(vector<int> &weights, int k) {
        int n = weights.size();
        if (n == k) {
            return 0;
        }
        long long int max_ans = 0;
        long long int min_ans = 0;
        for (int i = 1; i < n; ++i) {
            weights[i - 1] += weights[i];
        }
        std::sort(weights.begin(), weights.end() - 1);
        for (int i = 0; i < k - 1; ++i) {
            max_ans += weights[n - 2 - i];
            min_ans += weights[i];
        }
        return max_ans - min_ans;
    }

    int largestVariance(const string &s) {
        vector<int> nums(26, 0);
        for (auto c: s) ++nums[c - 'a'];
        int res = 0;
        for (char i = 'a'; i <= 'z'; ++i) {
            if (nums[i - 'a'] == 0) continue;
            for (char j = 'a'; j <= 'z'; ++j) {
                if (i == j || nums[j - 'a'] == 0) continue;
                int c1 = 0;
                int c2 = 0;
                for (auto iter = s.begin(); iter != s.end(); ++iter) {
                    if (*iter == i) ++c1;
                    if (*iter == j) ++c2;
                    if (c2 > c1) c1 = c2 = 0;
                    if (c2 && c1) res = max(res, c1 - c2);
                }
                c2 = c1 = 0;
                for (auto iter = s.rbegin(); iter != s.rend(); ++iter) {
                    if (*iter == i) ++c1;
                    if (*iter == j) ++c2;
                    if (c2 > c1) c1 = c2 = 0;
                    if (c2 && c1) res = max(res, c1 - c2);
                }
            }
        }
        return res;
    }

    int minDepth(TreeNode *root) {
        vector<TreeNode *> next;
        vector<TreeNode *> temp;
        if (!root) return 0;
        next.push_back(root);
        int res = 0;
        while (!next.empty()) {
            ++res;
            temp.clear();
            for (auto i: next) {
                if (i->left || i->right) {
                    if (i->left) temp.push_back(i->left);
                    if (i->right) temp.push_back(i->right);
                } else
                    return res;
            }
            temp.swap(next);
        }
        return res;
    }

    vector<int> distanceK(TreeNode *root, TreeNode *target, int k) {
        queue<TreeNode *> q;
        unordered_map<TreeNode *, TreeNode *> parent;
        q.push(root);

        while (!q.empty()) {
            TreeNode *front = q.front();
            q.pop();
            if (front->left) {
                parent[front->left] = front;
                q.push(front->left);
            }
            if (front->right) {
                parent[front->right] = front;
                q.push(front->right);
            }
        }

        unordered_map<TreeNode *, bool> visited;
        q.push(target);
        visited[target] = true;
        int cnt = 0;
        while (!q.empty()) {
            if (cnt == k) break;
            cnt++;
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode *front = q.front();
                q.pop();
                if (front->left && !visited[front->left]) {
                    q.push(front->left);
                    visited[front->left] = true;
                }
                if (front->right && !visited[front->right]) {
                    q.push(front->right);
                    visited[front->right] = true;
                }
                if (parent[front] && !visited[parent[front]]) {
                    q.push(parent[front]);
                    visited[parent[front]] = true;
                }
            }
        }
        vector<int> ans;
        while (!q.empty()) {
            TreeNode *front = q.front();
            q.pop();
            ans.push_back(front->val);
        }
        return ans;
    }

    vector<int> eventualSafeNodes(vector<vector<int>> &graph) {
        int n = graph.size();
        vector<short> isSafe(n, -1);
        auto f = [&](auto &&f, int node, unordered_set<int> &path) {
            if (isSafe[node] != -1) {
                return isSafe[node];
            }
            vector<int> &next = graph[node];
            if (next.empty())
                return isSafe[node] = 1;
            bool res = true;
            path.insert(node);
            for (auto i: next) {
                if (path.count(i)) {
                    path.erase(node);
                    return isSafe[node] = 0;
                }
                res = res && f(f, i, path);
            }
            path.erase(node);
            return isSafe[node] = res;
        };
        for (int i = 0; i < n; ++i) {
            unordered_set<int> visited;
            f(f, i, visited);
        }
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            if (isSafe[i]) {
                res.push_back(i);
            }
        }
        return std::move(res);
    }

    bool canFinish(int numCourses, vector<vector<int>> &prerequisites) {
        vector<bool> finished(numCourses, false);
        vector<bool> visited(numCourses, false);
        vector<vector<int>> graph(numCourses, vector<int>());

        for (auto &i: prerequisites) {
            graph[i[0]].emplace_back(i[1]);
        }
        auto f = [&](auto &&f, int course) -> bool {
            auto &next = graph[course];
            if (finished[course]) {
                return true;
            }
            if (visited[course]) {
                return false;
            }
            visited[course] = true;
            if (next.empty()) {
                return finished[course] = true;
            }
            for (auto i: next) {
                if (!f(f, i)) {
                    return finished[course] = false;
                }
            }
            return finished[course] = true;
        };
        for (int i = 0; i < numCourses; ++i) {
            if (!f(f, i)) {
                return false;
            }
        }
        return true;
    }

    int longestSubsequence(vector<int> &arr, int difference) {
        unordered_map<int, int> hm;
        int res = 0;
        for (auto i: arr) {
            hm[i] = max(hm[i - difference] + 1, hm[i]);
        }
        for (auto &k: hm) {
            res = max(res, k.second);
        }
        return res;
    }


    int maxValue(vector<vector<int>> &events, int k) {

        auto BS = [&](int index, vector<vector<int>> &events, int value) {
            int lo = index;
            int hi = events.size() - 1;
            int req = -1;
            while (hi >= lo) {
                int mid = lo + (hi - lo) / 2;
                if (events[mid][0] > value) {
                    req = mid;
                    hi = mid - 1;
                } else
                    lo = mid + 1;
            }
            return req;
        };

        vector<vector<int>> dp(events.size() + 5, vector<int>(k + 1, -1));
        auto helper = [&](auto &&helper, int i, int k) {
            if (i >= events.size() || k == 0 || i == -1) return 0;
            if (dp[i][k] != -1) return dp[i][k];

            int index = BS(i + 1, events, events[i][1]);

            int c1 = events[i][2] + helper(helper, index, k - 1);
            int c2 = helper(helper, i + 1, k);

            return dp[i][k] = max(c1, c2);
        };

        sort(events.begin(), events.end());
        int n = events.size();

        return helper(helper, 0, k);
    }

    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        stack<int> num1;
        stack<int> num2;
        num1.push(l1->val);
        num2.push(l2->val);
        while (l1->next) {
            num1.push(l1->next->val);
            l1 = l1->next;
        }
        while (l2->next) {
            num2.push(l2->next->val);
            l2 = l2->next;
        }
        bool carry = false;
        ListNode *res = nullptr;
        while (!num1.empty() || !num2.empty() || carry) {
            int c;
            if (!num1.empty() && !num2.empty()) {
                int a = num1.top();
                num1.pop();
                int b = num2.top();
                num2.pop();
                c = (a + b + carry) % 10;
                carry = (a + b + carry > 9);
            } else if (!num1.empty()) {
                int a = num1.top();
                num1.pop();
                c = (a + carry) % 10;
                carry = (a + carry > 9);
            } else if (!num2.empty()) {
                int b = num2.top();
                num2.pop();
                c = (b + carry) % 10;
                carry = (b + carry > 9);
            } else if (carry) {
                c = carry;
                carry = false;
            }
            if (res == nullptr) {
                res = new ListNode(c);
            } else {
                auto temp = new ListNode(c);
                temp->next = res;
                res = temp;
            }
        }
        return res;
    }

    int eraseOverlapIntervals(vector<vector<int>> &intervals) {
        sort(begin(intervals), end(intervals));
        int n = intervals.size();
        int res = 0;
        int end = INT32_MIN;
        for (int i = 0; i < n; ++i) {
            auto &interval = intervals[i];
            if (interval[0] >= end) {
                end = interval[1];
            } else {
                ++res;
                end = min(end, interval[1]);
            }
        }
        return res;
    }

    vector<int> asteroidCollision(vector<int> &asteroids) {
        vector<int> res;
        for (auto i: asteroids) {
            if (i > 0) {
                res.push_back(i);
            } else {
                bool flag = true;
                while (res.size() && res.back() > 0) {
                    if (res.back() > -i) {
                        flag = false;
                        break;
                    } else if (res.back() == -i) {
                        flag = false;
                        res.pop_back();
                        break;
                    } else if (res.back() > 0) {
                        res.pop_back();
                    } else {
                        break;
                    }
                }
                if (flag) res.push_back(i);
            }
        }
        return res;
    }

    int findNumberOfLIS(vector<int> &nums) {
        // dp[i] number of subseq ending at i
        // dp[x] = dp[i], dp[0]+1, dp[1]+1
        int n = nums.size();
        vector<int> length(n, 1);
        vector<int> count(n, 1);

        int longest = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[j] < nums[i]) {
                    if (length[j] + 1 > length[i]) {
                        count[i] = 0;
                        length[i] = 1 + length[j];
                    }

                    if (length[j] + 1 == length[i]) {
                        count[i] += count[j];
                    }
                }
            }

            longest = max(longest, length[i]);
        }

        int result = 0;
        for (int i = 0; i < n; ++i) {
            if (length[i] == longest) {
                result += count[i];
            }
        }
        return result;
    }

    double knightProbability(int n, int k, int row, int column) {
        constexpr int dx[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
        constexpr int dy[8] = {-1, 1, -2, 2, -2, 2, -1, 1};
        vector<vector<vector<double>>> dp(n + 1, vector<vector<double>>(n + 1, vector<double>(k + 1, -1.0)));
        auto f = [&](auto &&f, int x, int y, int k) -> double {
            if (x >= n || x < 0 || y >= n || y < 0) return 0.0;
            if (k <= 0) return 1.0;
            if (dp[x][y][k] != -1) return dp[x][y][k];
            dp[x][y][k] = 0.0;
            for (int i = 0; i < 8; ++i) {
                dp[x][y][k] += f(f, x + dx[i], y + dy[i], k - 1);
            }
            return dp[x][y][k];
        };

        return f(f, row, column, k) / ::pow(8.0, k);
    }

    vector<TreeNode *> allPossibleFBT(int n) {
        if ((n & 1) == 0) return {};
        int numNode = (n - 1) >> 1;
        vector<TreeNode *> res;
        TreeNode *root = new TreeNode(0);
        if (n == 1) {
            res.push_back(new TreeNode(0));
        } else {
            for (int i = 1; i <= n - 1; i += 2) {
                vector<TreeNode *> lTrees = allPossibleFBT(i);
                vector<TreeNode *> rTrees = allPossibleFBT(n - i - 1);
                for (TreeNode *lt: lTrees) {
                    for (TreeNode *rt: rTrees) {
                        res.push_back(new TreeNode(0, lt, rt));
                    }
                }
            }
        }
        return res;
    }

    int peakIndexInMountainArray(vector<int> &arr) {
        int low = 0;
        int high = arr.size() - 1;
        int mid = (low + high) >> 1;
        while (high > low) {
            mid = (low + high) >> 1;
            if (mid > 0 && mid < arr.size() - 1) {
                if (arr[mid] > arr[mid + 1] && arr[mid] > arr[mid - 1])
                    return mid;
                else if (arr[mid] < arr[mid + 1]) {
                    low = mid + 1;
                } else if (arr[mid] < arr[mid - 1]) {
                    high = mid - 1;
                }
            } else if (mid == 0) {
                if (arr[mid] > arr[mid + 1])
                    return mid;
                else {
                    low = mid + 1;
                }
            } else if (mid == arr.size() - 1) {
                if (arr[mid] > arr[mid - 1]) {
                    return mid;
                } else {
                    high = mid - 1;
                }
            }
        }
        return low;
    }

    int minSpeedOnTime(vector<int> &dist, double hour) {
        if (dist.size() > ::ceil(hour)) {
            return -1;
        }
        int high = INT32_MAX >> 1;
        int low = 1;
        auto f = [&](int speed) {
            double ans = 0;
            for (int i = 0; i < dist.size(); i++) {
                double d = dist[i] * 1.0 / speed;
                if (i != dist.size() - 1)
                    ans = ans + ceil(d);
                else
                    ans += d;
                if (ans > hour)
                    return false;
            }
            if (ans <= hour)
                return true;
            return false;
        };
        int i = 1;
        int j = 1e7;
        int minspeed = -1;
        while (i <= j) {
            int mid = i + (j - i) / 2;
            if (f(mid)) {
                minspeed = mid;
                j = mid - 1;
            } else
                i = mid + 1;
        }
        return minspeed;
    }

    int strangePrinter(const string &s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, -1));
        auto f = [&](auto &&f, int i, int j) {
            if (i > j || i < 0 || j < 0 || j >= n || i >= n) return 0;
            if (i == j) return 1;
            if (dp[i][j] != -1) return dp[i][j];
            dp[i][j] = f(f, i, j - 1) + 1;
            for (int k = i; k < j; ++k) {
                if (s[k] == s[j]) {
                    dp[i][j] = min(dp[i][j], f(f, i, k) + f(f, k + 1, j - 1));
                }
            }
            return dp[i][j];
        };
        return f(f, 0, n - 1);
    }

    double soupServings(int n) {
        map<pair<int, int>, double> cache;
        const int da[4] = {100, 75, 50, 25};
        const int db[4] = {0, 25, 50, 75};
        auto f = [&](auto &&f, int A, int B) {
            if (A < 0 || B < 0) return 0.0;
            if (cache.count({A, B})) {
                return cache[{A, B}];
            }
            double res = 0;
            for (int k = 0; k < 4; ++k) {
                int a = A - da[k];
                int b = B - db[k];
                if (a > 0 && b > 0) {
                    res += 0.25 * (f(f, a, b));
                } else if (a <= 0 && b <= 0) {
                    res += 0.25 * 0.5;
                } else if (a <= 0 && b > 0) {
                    res += 0.25;
                }
            }
            return cache[{A, B}] = res;
        };
        if (n >= 4800) return 1;// trick
        return f(f, n, n);
    }

    bool PredictTheWinner(vector<int> &nums) {
        int n = nums.size();
        vector<vector<int>> cache(n, vector<int>(n, -1));
        auto f = [&](auto &&f, int i, int j) {
            if (i < 0 || j < 0 || i >= n || j >= n || i > j) return 0;
            if (i == j) return nums[i];
            if (cache[i][j] != -1) return cache[i][j];
            return cache[i][j] = max(nums[i] - f(f, i + 1, j), nums[j] - f(f, i, j - 1));
        };
        return f(f, 0, n - 1) >= 0;
    }

    int minimumDeleteSum(const string &s1, const string &s2) {
        int n = s1.size();
        int m = s2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, INT32_MAX >> 2));
        dp[0][0] = 0;
        for (int j = 1; j < m; ++j) {
            dp[0][j] = dp[0][j - 1] + s2[j - 1];
        }
        for (int i = 1; i < n; ++i) {
            dp[i][0] = dp[i - 1][0] + s1[i - 1];
        }
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                dp[i][j] = min(dp[i][j - 1] + s2[j - 1], dp[i - 1][j] + s1[i - 1]);
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                }
            }
        }
        return dp[n][m];
    }

    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> combinations;
        vector<int> current;
        auto backtracking = [&](auto &&backtracking, int cur, int k, int n) -> void {
            if (current.size() == k) {
                combinations.push_back(current);
            } else {
                int i = 0;
                if (!current.empty()) {
                    i = *(current.end() - 1);
                }
                for (++i; i < n + 1; i++) {
                    current.push_back(i);
                    backtracking(backtracking, cur + 1, k, n);
                    current.pop_back();
                }
            }
        };
        backtracking(backtracking, 0, k, n);
        return combinations;
    }

    vector<vector<int>> permute(vector<int> &nums) {
        vector<vector<int>> res;
        std::sort(nums.begin(), nums.end());
        do {
            res.emplace_back(nums);
        } while (std::next_permutation(nums.begin(), nums.end()));
        return res;
    }

    long long maximumSubarraySum(vector<int> &nums, int k) {
        long long res = 0;
        unordered_map<int, int> times;
        long long sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += nums[i];
            ++times[nums[i]];
        }
        for (int i = k; i < nums.size(); ++i) {
            if (times.size() == k) {
                res = max(res, sum);
            }
            --times[nums[i - k]];
            if (times[nums[i - k]] == 0) {
                times.erase(nums[i - k]);
            }
            sum -= nums[i - k];
            sum += nums[i];
            ++times[nums[i]];
        }
        if (times.size() == k) {
            res = max(res, sum);
        }
        return res;
    }

    int takeCharacters(const string &s, int k) {
        std::array<int, 3> times = {0, 0, 0};
        int n = s.size();
        for_each(s.begin(), s.end(), [&times](auto i) {
            times[i - 'a']++;
        });
        if (std::any_of(times.begin(), times.end(), [k](auto i) {
                return i < k;
            })) return -1;
        int res = n;
        int i = 0;
        for (int j = 0; j < n; ++j) {
            times[s[j] - 'a']--;
            while (i < j && std::any_of(times.begin(), times.end(), [k](auto value) { return value < k; })) {
                ++times[s[i] - 'a'];
                ++i;
            }
            if (std::all_of(times.begin(), times.end(), [k](auto value) {
                    return value >= k;
                }))
                res = min(res, n - (j - i + 1));
        }
        return res;
    }

    int countUnguarded(int m, int n, vector<vector<int>> &guards, vector<vector<int>> &walls) {
        if (n != 1 && m != 1) {
            vector<vector<char>> grid(m, vector<char>(n, ' '));
            for (auto &i: walls) {
                grid[i[0]][i[1]] = 'w';
            }
            constexpr int dx[4] = {0, 0, 1, -1};
            constexpr int dy[4] = {1, -1, 0, 0};
            for (auto &i: guards) {
                int x = i.front();
                int y = i.back();
                grid[x][y] = 'g';
                for (int k = 0; k < 4; ++k) {
                    int xx = x + dx[k];
                    int yy = y + dy[k];
                    while (xx >= 0 && xx < m && yy >= 0 && yy < n) {
                        if (grid[xx][yy] == 'w')
                            break;
                        grid[xx][yy] = 'g';
                        xx = xx + dx[k];
                        yy = yy + dy[k];
                    }
                }
            }
            int res = 0;
            for (auto &i: grid) {
                for (auto j: i) {
                    res += (j == ' ' ? 1 : 0);
                }
            }
            return res;
        } else if (m == 1) {
            vector<char> grid(n, ' ');
            for (auto &i: walls) {
                grid[i[1]] = 'w';
            }
            constexpr int dx[2] = {1, -1};
            for (auto &i: guards) {
                int x = i.back();
                grid[x] = 'g';
                for (int k = 0; k < 2; ++k) {
                    int xx = x + dx[k];
                    while (xx >= 0 && xx < n) {
                        if (grid[xx] == 'w' || grid[xx] == 'g')
                            break;
                        grid[xx] = 'g';
                        xx = xx + dx[k];
                    }
                }
            }
            int res = 0;
            for (auto &i: grid) {
                res += (i == ' ' ? 1 : 0);
            }
            return res;
        } else {
            vector<char> grid(m, ' ');
            for (auto &i: walls) {
                grid[i[0]] = 'w';
            }
            constexpr int dx[2] = {1, -1};
            for (auto &i: guards) {
                int x = i.front();
                grid[x] = 'g';
                for (int k = 0; k < 2; ++k) {
                    int xx = x + dx[k];
                    while (xx >= 0 && xx < m) {
                        if (grid[xx] == 'w' || grid[xx] == 'g')
                            break;
                        grid[xx] = 'g';
                        xx = xx + dx[k];
                    }
                }
            }
            int res = 0;
            for (auto &i: grid) {
                res += (i == ' ' ? 1 : 0);
            }
            return res;
        }
    }

    int maxEqualRowsAfterFlips(vector<vector<int>> &matrix) {
        int m = matrix.size();
        int n = matrix.front().size();
        int res = 0;
        unordered_map<string, int> freq;
        for (auto &i: matrix) {
            string tmp(i.size(), '0');
            for (int j = 0; j < i.size(); ++j) {
                if (i[j] != i[0]) {
                    tmp[j] = '1';
                }
            }
            res = max(res, ++freq[tmp]);
        }
        return res;
    }

    vector<vector<char>> rotateTheBox(vector<vector<char>> &box) {
        int n = box.size();
        int m = box.begin()->size();
        vector<vector<char>> res(m, vector<char>(n, '.'));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (box[i][j] == '*') {
                    res[j][n - 1 - i] = '*';
                }
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = m - 1; j >= 0; --j) {
                if (box[i][j] == '#') {
                    int y = n - 1 - i;
                    int x = j;
                    while (x < m) {
                        if (x == m - 1 || res[x + 1][y] == '*' || res[x + 1][y] == '#') {
                            break;
                        } else {
                            ++x;
                        }
                    }
                    res[x][y] = '#';
                }
            }
        }
        return res;
    }

    long long maxMatrixSum(vector<vector<int>> &matrix) {
        int n = matrix.size();
        int prev_i = -1;
        int prev_j = -1;
        long long sum = 0ll;
        int min_pos_i = -1;
        int min_pos_j = -1;
        auto update_min_pos = [&](int x, int y) {
            if (min_pos_i == -1) {
                min_pos_i = x;
                min_pos_j = y;
            } else {
                if (matrix[x][y] < matrix[min_pos_i][min_pos_j]) {
                    min_pos_i = x;
                    min_pos_j = y;
                }
            }
        };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] < 0) {
                    if (prev_i == -1) {
                        prev_i = i;
                        prev_j = j;
                    } else {
                        matrix[i][j] = -matrix[i][j];
                        matrix[prev_i][prev_j] = -matrix[prev_i][prev_j];
                        update_min_pos(i, j);
                        update_min_pos(prev_i, prev_j);
                        sum += matrix[i][j] + matrix[prev_i][prev_j];
                        prev_i = -1;
                        prev_j = -1;
                    }
                } else {
                    update_min_pos(i, j);
                    sum += matrix[i][j];
                }
            }
        }
        if (prev_i != -1) {
            if (matrix[prev_i][prev_j] + matrix[min_pos_i][min_pos_j] < 0) {
                sum = sum - matrix[min_pos_i][min_pos_j];
                matrix[prev_i][prev_j] = -matrix[prev_i][prev_j];
                matrix[min_pos_i][min_pos_j] = -matrix[min_pos_i][min_pos_j];
                sum += matrix[min_pos_i][min_pos_j];
            }
            sum += matrix[prev_i][prev_j];
        }
        return sum;
    }

    int slidingPuzzle(vector<vector<int>> &board) {
        constexpr int directions[5] = {0, 1, 0, -1, 0};
        constexpr int base[6] = {100000, 10000, 1000, 100, 10, 1};

        auto encoding = [&]() {
            int res = 0;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    res += board[i][j] * base[i * 3 + j];
                }
            }
            return res;
        };

        auto decoding = [&](int num) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    board[i][j] = (num / base[i * 3 + j]) % 10;
                }
            }
        };
        auto swapZeroNeighbors = [&](int num) {
            int rows = 2, cols = 3;
            decoding(num);
            int x = -10;
            int y = -10;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (board[i][j] == 0) {
                        x = i;
                        y = j;
                        break;
                    }
                }
            }
            vector<int> results;
            for (int i = 0; i < 4; ++i) {
                int newX = x + directions[i];
                int newY = y + directions[i + 1];
                if (newX >= 0 && newX < rows && newY >= 0 && newY < cols) {
                    std::swap(board[x][y], board[newX][newY]);
                    results.push_back(encoding());
                    std::swap(board[x][y], board[newX][newY]);
                }
            }
            return results;
        };
        queue<int> q;
        queue<int> temp;
        unordered_set<int> seen;
        int res = 0;
        if (encoding() == 123450) {
            return 0;
        }
        q.emplace(encoding());
        while (q.size()) {
            temp.swap(q);
            ++res;
            while (temp.size()) {
                int state = temp.front();
                temp.pop();
                seen.emplace(state);
                for (auto i: swapZeroNeighbors(state)) {
                    if (i == 123450) {
                        return res;
                    }
                    if (seen.count(i)) {
                        continue;
                    } else {
                        q.emplace(i);
                    }
                }
            }
        }
        return -1;
    }

    int findChampion(int n, vector<vector<int>> &edges) {
        vector<bool> is_defeated(n, false);
        for (auto &i: edges) {
            is_defeated[i[1]] = true;
        }
        int res = -1;
        for (int i = 0; i < n; ++i) {
            if (!is_defeated[i] && res == -1) {
                res = i;
            } else if (!is_defeated[i]) {
                return -1;
            }
        }
        return res;
    }

    vector<int> shortestDistanceAfterQueries(int n, vector<vector<int>> &queries) {
        vector<vector<int>> next(n);
        for (int i = 0; i < n - 1; ++i) {
            next[i].push_back(i + 1);
        }
        vector<int> dp(n, -1);
        auto f = [&](auto &&f, int index) {
            int res = n - index;
            if (index == n - 1) return 0;
            if (dp[index] != -1) return dp[index];
            for (auto i: next[index]) {
                res = min(res, 1 + f(f, i));
            }
            return dp[index] = res;
        };
        vector<int> res;
        for (auto &i: queries) {
            next[i[0]].push_back(i[1]);
            std::fill(begin(dp), end(dp), -1);
            res.push_back(f(f, 0));
        }
        return res;
    }

    int minimumObstacles(vector<vector<int>> &grid) {
        int m = grid.size();
        int n = grid.front().size();
        queue<pair<int, int>> q;
        queue<pair<int, int>> tmp;
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        int res = 0;
        q.emplace(0, 0);
        constexpr int d[5] = {0, 1, 0, -1, 0};
        while (q.size()) {
            q.swap(tmp);
            while (tmp.size()) {
                auto [x, y] = tmp.front();
                tmp.pop();
                if (x == m - 1 && y == n - 1) return res;
                visited[x][y] = true;
                for (int i = 0; i < 4; ++i) {
                    int xx = x + d[i];
                    int yy = y + d[i + 1];
                    if (xx < 0 || xx >= m || yy < 0 || yy >= n) continue;
                    if (xx == m - 1 && yy == n - 1) return res;
                    if (!visited[xx][yy]) {
                        visited[xx][yy] = true;
                        if (grid[xx][yy] == 1) q.emplace(xx, yy);
                        else
                            tmp.emplace(xx, yy);
                    }
                }
            }
            ++res;
        }
        return -1;
    }

    int minimumTime(vector<vector<int>> &grid) {
        int m = grid.size();
        int n = grid.front().size();
        priority_queue<pair<int, pair<int, int>>> q;
        vector<vector<bool>> visited(m, vector<bool>(n, 0));
        q.emplace(0, pair<int, int>{0, 0});
        if (grid[0][1] > 1 && grid[1][0] > 1) return -1;
        constexpr int d[5] = {0, 1, 0, -1, 0};
        while (q.size()) {
            auto [time, coordinates] = q.top();
            auto [x, y] = coordinates;
            time = -time;
            q.pop();
            if (x == m - 1 && y == n - 1) return time;
            for (int i = 0; i < 4; ++i) {
                int xx = x + d[i];
                int yy = y + d[i + 1];
                if (xx < 0 || xx >= m || yy < 0 || yy >= n) continue;
                if (visited[xx][yy]) continue;
                visited[xx][yy] = true;
                int new_time = time + 1;
                if (grid[xx][yy] > time + 1)
                    new_time = grid[xx][yy] + ((grid[xx][yy] - time - 1) & 1);
                q.emplace(-new_time, pair<int, int>({xx, yy}));
            }
        }
        return -1;
    }

    vector<vector<int>> validArrangement(vector<vector<int>> &pairs) {
        unordered_map<int, vector<int>> adj;          // 邻接表
        unordered_map<int, int> in_degree, out_degree;// 入度和出度计数
        // 构建邻接表和度数
        for (auto &p: pairs) {
            adj[p[0]].push_back(p[1]);
            ++out_degree[p[0]];
            ++in_degree[p[1]];
        }
        // 找到起点
        int start = pairs[0][0];
        for (auto &[node, out_d]: out_degree) {
            if (out_d > in_degree[node]) {
                start = node;
                break;
            }
        }
        // 欧拉路径构建
        vector<vector<int>> result;
        stack<int> stack;
        stack.push(start);
        while (!stack.empty()) {
            int u = stack.top();
            if (!adj[u].empty()) {
                int v = adj[u].back();
                adj[u].pop_back();
                stack.push(v);
            } else {
                stack.pop();
                if (!stack.empty()) {
                    result.push_back({stack.top(), u});
                }
            }
        }
        reverse(result.begin(), result.end());
        return result;
    }

    bool checkIfExist(vector<int> &arr) {
        int n = arr.size();
        unordered_set<int> prev;
        for (auto i: arr) {
            if (prev.count(2 * i)) {
                return true;
            }
            prev.emplace(i);
            prev.emplace(i * 4);
        }
        return false;
    }

    int isPrefixOfWord(const string &sentence, const string &searchWord) {
        int index = 1;
        int index_target = 0;
        for (int i = 0; i < sentence.size(); ++i) {
            if (sentence[i] == ' ') {
                ++index;
                index_target = 0;
            } else {
                if (index_target == -1) {
                    continue;
                }
                if (searchWord[index_target] == sentence[i]) {
                    ++index_target;
                    if (index_target == searchWord.size()) {
                        return index;
                    }
                } else {
                    index_target = -1;
                }
            }
        }
        return -1;
    }

    bool canMakeSubsequence(const string &str1, const string &str2) {
        int index2 = 0;
        int index1 = 0;
        for (index1 = 0; index1 < str1.size(); ++index1) {
            if (str1[index1] == str2[index2] || (str1[index1] - 'a' + 1) % ('z' - 'a' + 1) + 'a' == str2[index2]) {
                ++index2;
                if (index2 == str2.size()) {
                    return true;
                }
            }
        }
        return false;
    }

    bool canChange(const string &start, const string &target) {
        int index1 = 0;
        int index2 = 0;
        assert(start.size() == target.size());
        int n = start.size();
        while (index1 < n && index2 < n) {
            while (index1 < n && start[index1] == '_')
                ++index1;
            while (index2 < n && target[index2] == '_')
                ++index2;
            if (start[index1] != target[index2])
                return false;
            if (start[index1] == 'L' && index1 < index2)
                return false;
            if (start[index1] == 'R' && index1 > index2)
                return false;
            ++index1;
            ++index2;
        }
        while (index1 < n) {
            if (start[index1++] != '_')
                return false;
        }
        while (index2 < n) {
            if (target[index2++] != '_')
                return false;
        }
        return true;
    }

    int maxCount(vector<int> &banned, int n, int maxSum) {
        int sum = 0;
        int res = 0;
        int index_banned = 0;
        std::sort(banned.begin(), banned.end());
        auto new_end = std::unique(banned.begin(), banned.end());
        for (int i = 1; i <= n; ++i) {
            if (index_banned < banned.size() && i == banned[index_banned]) {
                ++index_banned;
                continue;
            }
            sum += i;
            if (sum <= maxSum)
                ++res;
            else
                return res;
        }
        return res;
    }

    int minimumSize(vector<int> &nums, int maxOperations) {
        auto f = [&](int number) {
            int k = maxOperations;
            for (auto i: nums) {
                if (i > number) {
                    k -= (i - 1) / number;
                    if (k < 0) {
                        return false;
                    }
                }
            }
            return true;
        };
        int left = 1, right = *max_element(nums.begin(), nums.end());
        int res = right;
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (f(mid)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    int maximumLength(const string &s) {
        constexpr auto CHAR_NUM = 'z' - 'a' + 1;
        auto f = [&](int len) {
            vector<int> times(CHAR_NUM, 0);
            int prev = 0;
            int n = s.size();
            for (int i = 0; i < n; ++i) {
                while (s[i] != s[prev]) ++prev;
                if (i - prev + 1 >= len)
                    if (++times[s[i] - 'a'] >= 3) {
                        return true;
                    }
            }
            return false;
        };
        if (!f(1)) return -1;
        int left = 1;
        int res = 1;
        int right = s.size() - 1;
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (f(mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    int maximumBeauty(vector<int> &nums, int k) {
        std::sort(nums.begin(), nums.end());
        if (nums.back() - nums.front() <= 2 * k) return nums.size();
        int right = 0, n = nums.size();
        int res = 1;
        for (int left = 0; left < n; ++left) {
            while (std::abs(nums[right] - nums[left]) > 2 * k) ++right;
            res = max(res, left - right + 1);
        }
        return res;
    }

    long long pickGifts(vector<int> &gifts, int k) {
        priority_queue<int> pq(gifts.begin(), gifts.end());
        while (k--) {
            int top = pq.top();
            pq.pop();
            if (top == 1) {
                pq.push(1);
                break;
            } else {
                pq.push(sqrt(top));
            }
        }
        long long res = 0;
        while (pq.size()) {
            res += pq.top();
            pq.pop();
        }
        return res;
    }

    vector<bool> isArraySpecial(vector<int> &nums, vector<vector<int>> &queries) {
        int n = nums.size();
        if (n == 1) {
            return vector<bool>(queries.size(), true);
        }
        vector<bool> is_special(n, true);
        for (int i = 1; i < n; ++i) {
            int a = i - 1;
            int b = i;
            if (!((nums[a] + nums[b]) & 1)) {
                is_special[a] = false;
                is_special[b] = false;
            }
        }
        vector<int> prefix(n, 0);
        prefix[0] = is_special[0];
        for (int i = 1; i < n; ++i) {
            prefix[i] = prefix[i - 1];
            if (is_special[i])
                ++prefix[i];
        }
        vector<bool> res;
        res.reserve(queries.size());
        for (auto &i: queries) {
            int start = i[0];
            int end = i[1];
            if (start == end) {
                res.push_back(true);
            } else if (start == end - 1) {
                res.push_back(static_cast<bool>((nums[start] + nums[end]) & 1));
            } else {
                res.push_back(prefix[end - 1] - prefix[start] == end - start - 1);
            }
        }
        return res;
    }

    int maxTwoEvents(vector<vector<int>> &events) {
        using info = tuple<int, bool, int>;
        vector<info> times;
        times.reserve(events.size());
        for (auto &i: events) {
            times.emplace_back(i[0], false, i[2]);
            times.emplace_back(i[1], true, i[2]);
        }
        std::sort(times.begin(), times.end());
        int maxV = 0;
        int res = 0;
        for (auto [t, is_end, v]: times) {
            if (is_end) {
                maxV = max(maxV, v);
            } else {
                res = max(res, maxV + v);
            }
        }
        return res;
    }

    long long findScore(vector<int> &nums) {
        int n = nums.size();
        vector<bool> marked(n, false);
        priority_queue<pair<int, int>> pq;
        for (auto i = 0; i < nums.size(); ++i) {
            pq.emplace(-nums[i], -i);
        }
        long long res = 0;
        while (pq.size()) {
            auto [v, i] = pq.top();
            v = -v, i = -i;
            pq.pop();
            if (marked[i]) {
                continue;
            } else {
                res += v;
                marked[i] = true;
                if (i + 1 < n)
                    marked[i + 1] = true;
                if (i - 1 >= 0)
                    marked[i - 1] = true;
            }
        }
        return res;
    }

    long long continuousSubarrays(vector<int> &nums) {
        int left = 0;
        long long res = 0;
        priority_queue<pair<int, int>> maxs;
        priority_queue<pair<int, int>> mins;
        for (int right = 0; right < nums.size(); ++right) {
            int num = nums[right];
            maxs.emplace(num, -right);
            mins.emplace(-num, -right);
            while (maxs.top().first + mins.top().first > 2) {
                auto index_max = -maxs.top().second;
                auto index_min = -mins.top().second;
                left = min(index_max, index_min) + 1;
                while (-maxs.top().second < left) {
                    maxs.pop();
                }
                while (-mins.top().second < left) {
                    mins.pop();
                }
            }
            res += (right - left + 1);
        }
        return res;
    }

    double maxAverageRatio(vector<vector<int>> &classes, int extraStudents) {
        int n = classes.size();
        auto cmp = [](const pair<int, int> &left, const pair<int, int> &right) {
            double a = left.first, b = left.second;
            double c = right.first, d = right.second;
            return (a + 1) / (b + 1) - a / b < (c + 1) / (d + 1) - c / d;
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> mins(cmp);
        for (auto &i: classes) {
            mins.emplace(i[0], i[1]);
        }
        while (extraStudents--) {
            auto top = mins.top();
            mins.pop();
            mins.emplace(top.first + 1, top.second + 1);
        }
        double res = 0;
        while (mins.size()) {
            auto [upper, lower] = mins.top();
            mins.pop();
            res += static_cast<double>(upper) / lower;
        }
        return res / n;
    }

    vector<int> getFinalState(vector<int> &nums, int k, int multiplier) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        for (int i = 0; i < nums.size(); ++i) {
            pq.emplace(nums[i], i);
        }
        while (k--) {
            auto [v, index] = pq.top();
            pq.pop();
            pq.emplace(v * multiplier, index);
        }
        while (pq.size()) {
            auto [v, index] = pq.top();
            pq.pop();
            nums[index] = v;
        }
        return nums;
    }

    string repeatLimitedString(const string &s, int repeatLimit) {
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        vector<int> times(CHAR_NUM, 0);
        for (auto c: s) {
            times[c - 'a']++;
        }
        string res;
        int index = 'z' - 'a';
        auto find_next_index = [&]() {
            int res = index - 1;
            if (res < 0) {
                return -1;
            } else {
                while (res >= 0 && times[res] == 0) {
                    --res;
                }
                return res >= 0 ? res : -1;
            }
        };
        while (index >= 0) {
            int repeat = 0;
            while (times[index]) {
                int prev_len = 0;
                for (auto i = res.rbegin(); i != res.rend(); ++i) {
                    if (*i == index + 'a') {
                        ++prev_len;
                    } else
                        break;
                }
                int len = min(repeatLimit - prev_len, times[index]);
                times[index] -= len;
                res.append(string(len, 'a' + index));
                if (len + prev_len == repeatLimit) {
                    auto next_index = find_next_index();
                    if (next_index == -1) {
                        return res;
                    } else {
                        times[next_index]--;
                        res.push_back('a' + next_index);
                    }
                }
            }
            index = find_next_index();
        }
        return res;
    }

    vector<int> finalPrices(vector<int> &prices) {
        int n = prices.size();
        vector<pair<int, int>> q;
        for (int i = 0; i < n; ++i) {
            while (q.size() && q.back().first >= prices[i]) {
                prices[q.back().second] -= prices[i];
                q.pop_back();
            }
            q.emplace_back(prices[i], i);
        }
        return prices;
    }

    int maxChunksToSorted(vector<int> &arr) {
        int count = 0;
        int max_index = 0;
        for (int i = 0; i < arr.size(); ++i) {
            max_index = max(max_index, arr[i]);
            if (max_index == i) {
                ++count;
                max_index++;
            }
        }
        return count;
    }

    TreeNode *reverseOddLevels(TreeNode *root) {
        stack<int> vs;
        queue<TreeNode *> nodes;
        nodes.push(root);
        bool flag = true;
        queue<TreeNode *> tmp;
        while (nodes.size()) {
            auto front = nodes.front();
            if (!front->left) {
                return root;
            }
            if (flag) {
                while (nodes.size()) {
                    auto front = nodes.front();
                    nodes.pop();
                    vs.push(front->left->val);
                    tmp.push(front->left);
                    vs.push(front->right->val);
                    tmp.push(front->right);
                }
                while (tmp.size()) {
                    auto front = tmp.front();
                    tmp.pop();
                    nodes.push(front);
                    front->val = vs.top();
                    vs.pop();
                }
            } else {
                while (nodes.size()) {
                    auto front = nodes.front();
                    nodes.pop();
                    tmp.push(front->left);
                    tmp.push(front->right);
                }
                nodes.swap(tmp);
            }
            flag = !flag;
        }
        return root;
    }

    int maxKDivisibleComponents(int n, vector<vector<int>> &edges, vector<int> &values, int k) {
        long long res = 0;
        vector<vector<int>> next(n);
        vector<bool> visited(n, false);
        visited[0] = true;
        for (auto &i: edges) {
            next[i[0]].push_back(i[1]);
            next[i[1]].push_back(i[0]);
        }
        auto dfs = [&](auto &&dfs, int node) -> long long {
            auto &subtree = next[node];
            long long sum = values[node];
            for (auto i: subtree) {
                if (visited[i])
                    continue;
                visited[i] = true;
                long long subtree_sum = dfs(dfs, i);
                if (subtree_sum % k == 0) {
                    ++res;
                } else
                    sum += subtree_sum;
            }
            return sum;
        };
        res += dfs(dfs, 0) % k == 0;
        return res;
    }

    int minimumOperations(TreeNode *root) {
        int res = 0;
        deque<TreeNode *> q;
        deque<TreeNode *> tmp;
        vector<int> vals;
        q.push_back(root);
        while (q.size()) {
            while (q.size()) {
                auto node = q.front();
                q.pop_front();
                vals.push_back(node->val);
                if (node->left) {
                    tmp.push_back(node->left);
                }
                if (node->right) {
                    tmp.push_back(node->right);
                }
            }
            vector<pair<int, int>> sorted;
            for (int i = 0; i < vals.size(); ++i) {
                sorted.emplace_back(vals[i], i);
            }
            std::sort(sorted.begin(), sorted.end());
            vector<bool> visited(vals.size(), false);
            for (int i = 0; i < vals.size(); ++i) {
                if (visited[i]) continue;
                visited[i] = true;
                int start = i;
                int next = sorted[start].second;
                visited[next] = true;
                while (start != next) {
                    ++res;
                    start = next;
                    next = sorted[start].second;
                    if (visited[next])
                        break;
                    visited[next] = true;
                }
            }
            tmp.swap(q);
            vals.clear();
        }
        return res;
    }

    vector<int> largestValues(TreeNode *root) {
        vector<int> res;
        auto dfs = [&](auto &&dfs, TreeNode *node, int level) -> void {
            if (node) {
                if (res.size() < level + 1) {
                    res.push_back(node->val);
                } else {
                    res[level] = max(res[level], node->val);
                }
                dfs(dfs, node->left, level + 1);
                dfs(dfs, node->right, level + 1);
            }
        };
        dfs(dfs, root, 0);
        return res;
    }

    int minimumDiameterAfterMerge(vector<vector<int>> &edges1, vector<vector<int>> &edges2) {
        auto bfs = [&](int start, const vector<vector<int>> &adj) -> pair<int, int> {
            int n = adj.size();
            vector<int> dist(n, -1);
            queue<int> q;
            q.push(start);
            dist[start] = 0;
            int farthest_node = start;
            int max_dist = 0;

            while (!q.empty()) {
                int node = q.front();
                q.pop();
                for (auto &neighbor: adj[node]) {
                    if (dist[neighbor] == -1) {
                        dist[neighbor] = dist[node] + 1;
                        q.push(neighbor);
                        if (dist[neighbor] > max_dist) {
                            max_dist = dist[neighbor];
                            farthest_node = neighbor;
                        }
                    }
                }
            }
            return {farthest_node, max_dist};
        };
        // Function to compute the diameter of a tree using double BFS
        auto treeDiameter = [&](const vector<vector<int>> &adj) -> int {
            // First BFS to find one end of the diameter
            pair<int, int> first = bfs(0, adj);
            // Second BFS from the farthest node found in the first BFS
            pair<int, int> second = bfs(first.first, adj);
            return second.second;
        };
        // Number of nodes in each tree
        int n = edges1.size() + 1;
        int m = edges2.size() + 1;

        // Build adjacency lists
        vector<vector<int>> adj1(n, vector<int>());
        vector<vector<int>> adj2(m, vector<int>());
        for (auto &edge: edges1) {
            adj1[edge[0]].push_back(edge[1]);
            adj1[edge[1]].push_back(edge[0]);
        }
        for (auto &edge: edges2) {
            adj2[edge[0]].push_back(edge[1]);
            adj2[edge[1]].push_back(edge[0]);
        }

        // Compute diameters of both trees
        int diameter1 = treeDiameter(adj1);
        int diameter2 = treeDiameter(adj2);

        // To minimize the diameter after merging, connect the centers of both trees
        // However, for simplicity, we can consider that the new diameter is the maximum of:
        // - diameter1
        // - diameter2
        // - (ceil(diameter1/2) + ceil(diameter2/2) + 1)
        // This is based on the property that connecting two trees via their centers minimizes the diameter.

        // Find the radius (ceil(diameter / 2)) of both trees
        int radius1 = (diameter1 + 1) / 2;
        int radius2 = (diameter2 + 1) / 2;

        // The new diameter after merging
        int new_diameter = max({diameter1, diameter2, radius1 + radius2 + 1});

        return new_diameter;
    }

    int findTargetSumWays(vector<int> &nums, int target) {
        int n = nums.size();
        int res = 0;
        int sum = std::accumulate(nums.begin(), nums.end(), 0);
        if ((sum + target) % 2 != 0) return 0;
        if (abs(target) > abs(sum)) return 0;
        unordered_map<int, int> results_times;
        unordered_map<int, int> tmp;
        results_times[0] = 1;
        for (auto i: nums) {
            for (auto &k: results_times) {
                tmp[k.first + i] += k.second;
                tmp[k.first - i] += k.second;
            }
            results_times.swap(tmp);
            tmp.clear();
        }
        return results_times[target];
    }

    int maxScoreSightseeingPair(vector<int> &values) {
        auto n = values.size();
        int max_prev = values[0];
        int res = 0;
        for (int i = 1; i < n; ++i) {
            res = max(res, max_prev + values[i] - i);
            max_prev = max(max_prev, values[i] + i);
        };
        return res;
    }

    vector<int> leftmostBuildingQueries(vector<int> &heights, vector<vector<int>> &queries) {
        auto numberOfHeights = heights.size();
        auto numberOfQueries = queries.size();
        vector<int> indices;
        vector<int> res(numberOfQueries);
        for (auto i = 0; i < numberOfQueries; ++i) {
            auto &a = queries[i][0];
            auto &b = queries[i][1];
            if (a > b) std::swap(a, b);
            if (a == b || heights[b] > heights[a]) {
                res[i] = b;
                continue;
            }
            indices.emplace_back(i);
        }
        std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
            return queries[a][1] < queries[b][1];
        });
        vector<int> s;
        int right = heights.size() - 1;
        for (int i = indices.size() - 1; i >= 0; --i) {
            auto index = indices[i];
            auto a = queries[index][0];
            auto b = queries[index][1];
            while (right > b) {
                while (s.size() && heights[s.back()] <= heights[right]) {
                    s.pop_back();
                }
                s.emplace_back(right);
                --right;
            }
            auto iter = std::upper_bound(s.rbegin(), s.rend(), a, [&](auto x, auto y) {
                return heights[x] < heights[y];
            });
            if (iter == s.rend()) {
                res[index] = -1;
            } else {
                res[index] = *iter;
            }
        }
        return res;
    }

    int numWays(vector<string> &words, const string &target) {
        int numberOfWords = words.size();
        int lengthOfTarget = target.size();
        int lengthOfWord = words.back().size();
        constexpr int MOD = 1e9 + 7;
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        int res = 0;
        vector<vector<int>> chars(lengthOfWord, vector<int>(CHAR_NUM, 0));
        for (auto &i: words) {
            for (auto j = 0; j < i.size(); ++j) {
                chars[j][i[j] - 'a']++;
            }
        }
        vector<vector<int>> dp(lengthOfWord, vector<int>(lengthOfTarget, -1));
        auto dfs = [&](auto &&dfs, int k, int i) -> long long {
            if (i >= lengthOfTarget) {
                return 1;
            }
            if (k == lengthOfWord || lengthOfTarget - i > lengthOfWord - k) {
                // No valid solution, we should return 0 not -1;
                return 0;
            }
            if (dp[k][i] != -1)
                return dp[k][i];
            dp[k][i] = dfs(dfs, k + 1, i) % MOD;
            char c = target[i];
            if (chars[k][c - 'a'] > 0) {
                auto tmp = dfs(dfs, k + 1, i + 1);
                dp[k][i] = (dp[k][i] + chars[k][c - 'a'] * tmp % MOD) % MOD;
            }
            return dp[k][i] % MOD;
        };
        return dfs(dfs, 0, 0);
    }

    int countGoodStrings(int low, int high, int zero, int one) {
        vector<int> dp(high + 1, -1);
        int res = 0;
        dp[0] = 1;
        constexpr int MOD = 1e9 + 7;
        auto f = [&](auto &&f, int len) -> long long {
            if (len > high || len < 0) return 0;
            if (dp[len] != -1)
                return dp[len];
            return dp[len] = (f(f, len - zero) + f(f, len - one)) % MOD;
        };
        for (int i = low; i <= high; ++i) {
            res = (f(f, i) + res) % MOD;
        }
        return res;
    }

    int mincostTickets(vector<int> &days, vector<int> &costs) {
        int n = days.size();
        vector<int> dp(n + 1, std::numeric_limits<int>::max());
        constexpr int len[3] = {1, 7, 30};
        dp[0] = 0;
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) {
                int cost = dp[i] + costs[k];
                dp[i + 1] = min(dp[i + 1], cost);
                for (int j = i + 1; j < n; ++j) {
                    if (days[j] < days[i] + len[k]) {
                        dp[j + 1] = min(dp[j + 1], cost);
                    } else
                        break;
                }
            }
        }
        return dp.back();
    }

    int maxScore(const string &s) {
        int n = s.size();
        int ones = 0;
        int zeros = 0;
        for (auto i: s) {
            if (i == '1') ++ones;
        }
        int res = ones - 1;
        for (int i = 0; i < n - 1; ++i) {
            if (s[i] == '1') {
                --ones;
            } else
                ++zeros;
            res = max(res, zeros + ones);
        }
        return res;
    }

    vector<int> vowelStrings(vector<string> &words, vector<vector<int>> &queries) {
        int n = words.size();
        vector<int> prefix(n + 1, 0);
        auto is_vowel = [&](char c) {
            return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
        };
        for (int i = 0; i < n; ++i) {
            auto &s = words[i];
            if (is_vowel(s.front()) && is_vowel(s.back())) {
                prefix[i + 1] = prefix[i] + 1;
            } else {
                prefix[i + 1] = prefix[i];
            }
        }
        vector<int> res(queries.size());
        for (int i = 0; i < queries.size(); ++i) {
            res[i] = prefix[queries[i].back() + 1] - prefix[queries[i].front()];
        }
        return res;
    }

    int waysToSplitArray(vector<int> &nums) {
        int n = nums.size();
        vector<long long int> prefix(n);
        prefix[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            prefix[i] = prefix[i - 1] + nums[i];
        }
        int res = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (prefix[i] >= prefix[n - 1] - prefix[i]) {
                ++res;
            }
        }
        return res;
    }

    int countPalindromicSubsequence(const string &s) {
        int n = s.size();
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        vector<vector<int>> prefix(CHAR_NUM);
        vector<std::bitset<CHAR_NUM>> result(CHAR_NUM, 0);
        vector<int> nums(CHAR_NUM, 0);
        for (int i = 0; i < s.size(); ++i) {
            int char_index = s[i] - 'a';
            if (prefix[char_index].empty()) {
                ++nums[char_index];
                prefix[char_index] = nums;
                continue;
            } else {
                for (int k = 0; k < CHAR_NUM; ++k) {
                    if (nums[k] - prefix[char_index][k] > 0) {
                        result[char_index].set(k);
                    }
                }
            }
            ++nums[char_index];
        }
        int res = 0;
        for (auto &i: result)
            res += i.count();
        return res;
    }

    string shiftingLetters(const string &s, vector<vector<int>> &shifts) {
        int n = s.size();
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        vector<int> delta(n + 1);
        for (auto &i: shifts) {
            if (i[2] == 1) {
                ++delta[i[0]];
                delta[i[1] + 1] = (CHAR_NUM + delta[i[1] + 1] - 1) % CHAR_NUM;
            } else {
                delta[i[0]] = (CHAR_NUM + delta[i[0]] - 1) % CHAR_NUM;
                ++delta[i[1] + 1];
            }
        }
        string res(n, ' ');
        int delta_sum = 0;
        for (int i = 0; i < n; ++i) {
            delta_sum = (delta_sum + delta[i]) % CHAR_NUM;
            res[i] = (s[i] - 'a' + delta_sum) % CHAR_NUM + 'a';
        }
        return res;
    }

    vector<int> minOperations(const string &boxes) {
        int n = boxes.size();
        int prefix = (boxes[0] == '1');
        int sum = 0;
        for (int i = 0; i < boxes.size(); ++i) {
            sum += (boxes[i] == '1');
        }
        vector<int> res(n);
        int initial = 0;
        for (int i = 1; i < n; ++i) {
            if (boxes[i] == '1') {
                initial += i;
            }
        }
        res[0] = initial;
        for (int i = 1; i < n; ++i) {
            res[i] = (initial = (initial - (sum - prefix) + prefix));
            prefix += (boxes[i] == '1');
        }
        return std::move(res);
    }

    int countPrefixSuffixPairs(vector<string> &words) {
        int n = words.size();
        auto is_prefix_suffix = [](const string &a, const string &b) {
            auto bi = b.begin();
            for (auto i = a.begin(); i != a.end(); ++i) {
                if (*i != *(bi++)) return false;
            }
            auto rbi = b.rbegin();
            for (auto i = a.rbegin(); i != a.rend(); ++i) {
                if (*i != *(rbi++)) return false;
            }
            return true;
        };
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (words[i].size() <= words[j].size()) {
                    if (is_prefix_suffix(words[i], words[j])) ++res;
                }
            }
        }
        return res;
    }

    vector<string> stringMatching(vector<string> &words) {
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        struct TrieNode {
            int times = 0;
            vector<TrieNode *> nexts;
        } root;
        root.nexts = vector<TrieNode *>(CHAR_NUM);
        for (auto &i: words) {
            vector<TrieNode *> tmp;
            vector<TrieNode *> next;
            tmp.push_back(&root);
            for (auto c: i) {
                int char_index = c - 'a';
                for (auto node: tmp) {
                    if (node->nexts[char_index] != nullptr) {
                        node->nexts[char_index]->times++;
                        next.push_back(node->nexts[char_index]);
                    } else {
                        node->nexts[char_index] = new TrieNode;
                        node->nexts[char_index]->nexts = vector<TrieNode *>(CHAR_NUM);
                        node->nexts[char_index]->times++;
                        next.push_back(node->nexts[char_index]);
                    }
                }
                tmp.swap(next);
                next.clear();
                tmp.push_back(&root);
            }
        }
        vector<string> res;
        for (auto &i: words) {
            auto node = &root;
            bool flag = true;
            for (auto c: i) {
                int char_index = c - 'a';
                if (node->nexts[char_index] && node->nexts[char_index]->times > 1)
                    node = node->nexts[char_index];
                else {
                    flag = false;
                    break;
                }
            }
            if (flag) res.push_back(i);
        }
        return res;
    }

    int prefixCount(vector<string> &words, const string &pref) {
        int res = 0;
        auto is_pref = [](const string &a, const string &b) {
            auto bi = b.begin();
            for (auto iter = a.begin(); iter != a.end(); ++iter) {
                if (*(bi++) != *iter) return false;
            }
            return true;
        };
        for (auto &i: words) {
            if (is_pref(pref, i)) ++res;
        }
        return res;
    }

    vector<string> wordSubsets(vector<string> &words1, vector<string> &words2) {
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        vector<int> minimum(CHAR_NUM, 0);
        vector<int> tmp(CHAR_NUM, 0);
        for (auto &i: words2) {
            std::fill(tmp.begin(), tmp.end(), 0);
            for (auto c: i) {
                ++tmp[c - 'a'];
            }
            for (int k = 0; k < CHAR_NUM; ++k) {
                minimum[k] = max(minimum[k], tmp[k]);
            }
        }
        vector<string> res;
        for (auto &i: words1) {
            std::fill(tmp.begin(), tmp.end(), 0);
            for (auto c: i) {
                ++tmp[c - 'a'];
            }
            bool flag = true;
            for (int k = 0; k < CHAR_NUM; ++k) {
                if (tmp[k] < minimum[k]) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                res.emplace_back(i);
        }
        return res;
    }

    bool canConstruct(const string &s, int k) {
        if (s.size() < k) return false;
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        vector<int> nums(CHAR_NUM, 0);
        for (auto c: s) {
            int char_index = c - 'a';
            ++nums[char_index];
        }
        int odd_num = 0;
        for (auto i: nums) {
            if (i & 1) {
                ++odd_num;
            }
        };
        if (odd_num > k) return false;
        return true;
    }

    int minimumLength(const string &s) {
        constexpr int CHAR_NUM = 'z' - 'a' + 1;
        vector<int> nums(CHAR_NUM, 0);
        for (auto i: s) {
            int char_index = i - 'a';
            ++nums[char_index];
            if (nums[char_index] == 3) {
                nums[char_index] = 1;
            }
        }
        return std::accumulate(nums.begin(), nums.end(), 0);
    }

    bool canBeValid(const string &s, const string &locked) {
        //从左到右，从右到左，2次遍历，分别检查'('、')'是否满足要求。
        int remains = 0;
        int n = s.size();
        if (n & 1) return false;
        for (int i = 0; i < n; ++i) {
            if (locked[i] == '1') {
                if (s[i] == '(') {
                    ++remains;
                } else {
                    --remains;
                    if (remains < 0)
                        return false;
                }
            } else {
                ++remains;
            }
        }
        remains = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (locked[i] == '1') {
                if (s[i] == ')') {
                    ++remains;
                } else {
                    --remains;
                    if (remains < 0)
                        return false;
                }
            } else {
                ++remains;
            }
        }
        return true;
    }

    vector<int> findThePrefixCommonArray(vector<int> &A, vector<int> &B) {
        int n = A.size();
        constexpr int SIZE = 51;
        std::bitset<SIZE> a_bit(0);
        std::bitset<SIZE> b_bit(0);
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            a_bit.set(A[i]);
            b_bit.set(B[i]);
            std::bitset<SIZE> c_bit(0);
            for (int k = 0; k < SIZE; ++k) {
                if (a_bit.test(k) && b_bit.test(k)) {
                    c_bit.set(k);
                }
            }
            res.push_back(c_bit.count());
        }
        return res;
    }

    int minimizeXor(int num1, int num2) {
        constexpr int SIZE = 32;
        std::bitset<SIZE> num2_bits(num2);
        std::bitset<SIZE> num1_bits(num1);
        int set_bit_num = num2_bits.count();
        int num1_set_bit_num = num1_bits.count();
        if (num1_set_bit_num == set_bit_num) return num1;
        if (num1_set_bit_num > set_bit_num) {
            int remains = set_bit_num;
            for (int k = SIZE - 1; k >= 0; --k) {
                if (num1_bits.test(k)) {
                    --remains;
                    num1_bits.flip(k);
                    if (remains <= 0) return num1_bits.to_ulong() ^ num1;
                }
            }
        } else {
            vector<bool> flipped(SIZE, false);
            int remains = set_bit_num;
            for (int k = SIZE - 1; k >= 0; --k) {
                if (num1_bits.test(k)) {
                    --remains;
                    num1_bits.flip(k);
                    flipped[k] = true;
                }
            }
            for (int k = 0; k < SIZE; ++k) {
                if (!flipped[k]) {
                    num1_bits.flip(k);
                    --remains;
                    flipped[k] = true;
                    if (remains <= 0) return num1_bits.to_ulong() ^ num1;
                }
            }
        }
        return num1_bits.to_ulong() ^ num1;
    }

    int xorAllNums(vector<int> &nums1, vector<int> &nums2) {
        int res = 0;
        int n2 = nums2.size();
        int n1 = nums1.size();
        if (n1 & 1) {
            for (auto i: nums2) {
                res ^= i;
            }
        }
        if (n2 & 1) {
            for (auto i: nums1) {
                res ^= i;
            }
        }
        return res;
    }
};

int main() {
    Solution s;
    vector<int> a = {18, 64, 12, 21, 21, 78, 36, 58, 88, 58, 99, 26, 92, 91, 53, 10, 24, 25, 20, 92, 73, 63, 51, 65, 87,
                     6, 17, 32, 14, 42, 46, 65, 43, 9, 75};
    cout << s.totalCost(a, 13, 23) << endl;
    return 0;
}