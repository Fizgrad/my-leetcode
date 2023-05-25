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

};

int main() {
    Solution s;
    return 0;
}