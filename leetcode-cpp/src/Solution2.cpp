//
// Created by David Chen on 5/20/23.
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

class Solution {
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
                    static_cast<double>(bombs[i][2]) * bombs[i][2] <= 0) {
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

};

int main() {
    Solution s;
    return 0;
}