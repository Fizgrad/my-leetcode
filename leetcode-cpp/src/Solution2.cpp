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

};

int main() {
    Solution s;
    return 0;
}