//
// Created by Fitzgerald on 2/19/23.
//
#include<iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <array>
#include <deque>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <numeric>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    TreeNode *invertTree(TreeNode *root) {
        if (root != nullptr) {
            TreeNode *temp = root->left;
            root->left = root->right;
            root->right = temp;
            invertTree(root->right);
            invertTree(root->left);
        }
        return root;
    }

    double maxAverageRatio(vector<vector<int>> &classes, int extraStudents) {
        auto profit = [](double pass, double total) {
            return (pass + 1) / (total + 1) - pass / total;
        };
        double total = 0;
        priority_queue<pair<double, array<int, 2>>> pq;
        for (auto &c: classes) {
            total += (double) c[0] / c[1];
            pq.push({profit(c[0], c[1]), {c[0], c[1]}});
        }
        while (extraStudents--) {
            auto i = pq.top();
            pq.pop();
            total += i.first;
            pq.push({profit(i.second[0] + 1, i.second[1] + 1), {i.second[0] + 1, i.second[1] + 1}});
        }
        return total / classes.size();
    }

    vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
        deque<TreeNode *> qu;
        vector<vector<int>> res;
        qu.push_back(root);
        int need_reverse = 1;
        while (!qu.empty()) {
            need_reverse = 1 - need_reverse;
            vector<int> layer;
            deque<TreeNode *> temp;
            TreeNode *top = qu.front();
            while (!qu.empty()) {
                if (top != nullptr) {
                    temp.push_back(top->left);
                    temp.push_back(top->right);
                    layer.push_back(top->val);
                }
                qu.pop_front();
                top = qu.front();
            }
            while (!temp.empty()) {
                qu.push_back(temp.front());
                temp.pop_front();
            }
            if (need_reverse) {
                reverse(layer.begin(), layer.end());
            }
            if (!layer.empty())
                res.push_back(layer);
        }
        return res;
    }

    int searchInsert(vector<int> &nums, int target) {
        int i = 0;
        int j = nums.size();
        int mid;
        if (target > nums[j - 1]) {
            return j;
        }
        while (i <= j) {
            mid = (i + j) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                j = mid - 1;
            } else {
                i = mid + 1;
            }
        }
        return i;
    }

    string bestHand(vector<int> &ranks, vector<char> &suits) {
        unordered_map<int, int> s1;
        unordered_set<char> s2;
        for (int i = 0; i < ranks.size(); i++) {
            s1[ranks[i]]++;
            s2.insert(suits[i]);
        }
        if (s2.size() == 1)
            return "Flush";
        int c = 0;
        for (auto x: s1) {
            if (c < x.second)
                c = x.second;
        }
        switch (c) {
            case 1:
                return "High Card";
            case 2:
                return "Pair";
            case 3:
                return "Three of a Kind";
            case 4:
                return "Three of a Kind";
        }
        return "";
    }

    int strStr(string haystack, string needle) {
        vector<vector<int>> dfa(256, vector<int>(needle.size()));
        dfa[needle[0]][0] = 1;
        for (int x = 0, j = 1; j < needle.size(); ++j) {
            for (int c = 0; c < 256; ++c) {
                dfa[c][j] = dfa[c][x];
            }
            dfa[needle[j]][j] = j + 1;
            x = dfa[needle[j]][x];
        }
        int i = 0, j = 0;
        for (; i < haystack.size() && j < needle.size(); ++i) {
            j = dfa[haystack[i]][j];
        }
        if (j == needle.size())
            return i - needle.size();
        else return -1;
    }

    int strStr2(string &haystack, const string &needle) {
        return haystack.find(needle);
    }

    long long countSubarrays(vector<int> &nums, int minK, int maxK) {
        long res = 0;
        bool minFound = false, maxFound = false;
        int start = 0, minStart = 0, maxStart = 0;
        for (int i = 0; i < nums.size(); i++) {
            int num = nums[i];
            if (num < minK || num > maxK) {
                minFound = false;
                maxFound = false;
                start = i + 1;
            }
            if (num == minK) {
                minFound = true;
                minStart = i;
            }
            if (num == maxK) {
                maxFound = true;
                maxStart = i;
            }
            if (minFound && maxFound) {
                res += (min(minStart, maxStart) - start + 1);
            }
        }
        return res;
    }

    int countTriplets(vector<int> &nums) {
        int cnt = 0;
        int tuples[1 << 16] = {};
        for (auto a: nums)
            for (auto b: nums) ++tuples[a & b];
        for (auto a: nums)
            for (auto i = 0; i < (1 << 16); ++i)
                if ((i & a) == 0) cnt += tuples[i];
        return cnt;
    }

    int minJumps(vector<int> &arr) {
        unordered_map<int, vector<int>> hm;
        for (int i = 0; i < arr.size(); ++i) {
            hm[arr[i]].push_back(i);
        }
        queue<int> q;
        q.push(0);
        int res = 0;
        vector<int> visited(arr.size(), 0);
        visited[0] = 1;
        while (!q.empty()) {
            queue<int> temp;
            while (!q.empty()) {
                int front = q.front();
                if (front == arr.size() - 1) {
                    return res;
                }
                if (front - 1 >= 0 && !visited[front - 1]) {
                    temp.push(front - 1);
                    visited[front - 1] = 1;
                }
                if (front + 1 < arr.size() && !visited[front + 1]) {
                    temp.push(front + 1);
                    visited[front + 1] = 1;
                }
                for (auto i: hm[arr[front]]) {
                    if (!visited[i]) {
                        temp.push(i);
                        visited[i] = 1;
                    }
                }
                // clear() is very important, since after once you iterate over this vector, its content is meaningless
                // but it still use a quantity of time to check it, so we must clear it to get better performance.
                hm[arr[front]].clear();
                q.pop();
            }
            ++res;
            while (!temp.empty()) {
                q.push(temp.front());
                temp.pop();
            }
        }
        return res;
    }

    int minOperationsMaxProfit(vector<int> &customers, int boardingCost, int runningCost) {
        int max_profit = 0;
        int res = -1;
        int waiting = 0;
        int cur_profit = 0;
        int i = 0;
        while (i < customers.size() || waiting > 0) {
            if (i < customers.size())
                waiting += customers[i];
            int boarding = min(waiting, 4);
            waiting -= boarding;
            ++i;
            cur_profit += boarding * boardingCost - runningCost;
            if (cur_profit > max_profit) {
                max_profit = cur_profit;
                res = i;
            }
        }
        return res;
    }

    int findKthPositive(vector<int> &arr, int k) {
        int index = 0;
        int i;
        for (i = 1; i <= arr[arr.size() - 1]; ++i) {
            if (i == arr[index]) {
                ++index;
            } else {
                --k;
                if (k == 0) {
                    return i;
                }
            }
        }
        while (--k) {
            ++i;
        }
        return i;
    }

    int minimumDeletions(const string &s) {
        int cur_del = 0;
        int b = 0;
        int a = 0;
        for (auto i: s) {
            if (i == 'a') {
                if (b == 0) {
                    continue;
                } else {
                    a++;
                }
            } else {
                ++b;
            }
            if (a >= b) {
                cur_del += b;
                b = 0;
                a = 0;
            }
        }
        return cur_del + a;
    }

    long long kthLargestLevelSum(TreeNode *root, int k) {
        queue<TreeNode *> q;
        q.push(root);
        priority_queue<long long> res;
        while (!q.empty()) {
            queue<TreeNode *> temp;
            long long sum_val = 0;
            while (!q.empty()) {
                TreeNode *top = q.front();
                q.pop();
                if (top != nullptr) {
                    sum_val += top->val;
                    if (top->left)
                        temp.push(top->left);
                    if (top->right)
                        temp.push(top->right);
                }
            }

            res.push(-sum_val);
            if (res.size() > k) {
                res.pop();
            }
            q = temp;

        }
        return res.size() == k ? -res.top() : -1;
    }

    bool minimumTimeHelper(vector<int> &times, long long totalTrips, long long curTimes) {
        long long curTrips = 0;
        for (auto i: times) {
            curTrips += curTimes / i;
        }
        return curTrips >= totalTrips;
    }

    long long minimumTime(vector<int> &times, int totalTrips) {
        int n = times.size();
        if (n == 1) {
            return (long long) totalTrips * times[0];
        }
        sort(begin(times), end(times));
        long long max_num = (long long) totalTrips * (*(times.begin() + n - 1));
        long long min_num = (long long) totalTrips * (*times.begin()) / n;
        long long mid = (max_num + min_num) / 2;
        while (min_num < max_num) {
            bool flag = minimumTimeHelper(times, totalTrips, mid);
            if (flag) {
                max_num = mid;
            } else {
                min_num = mid + 1;
            }
            mid = (max_num + min_num) / 2;
        }
        return max_num;
    }

    vector<string> multiply(vector<string> &a, vector<string> &b) {
        vector<string> res;
        if (a.empty()) {
            return b;
        }
        if (b.empty()) {
            return a;
        }
        for (auto &i: a) {
            for (auto &j: b) {
                res.push_back(i + j);
            }
        }
        return res;
    }

    vector<string> parseBrace(string::iterator begin, string::iterator end) {
        vector<string> res;
        vector<string> temp;
        for (auto i = begin; i != end; ++i) {
            if (isalpha(*i)) {
                string a(i, i + 1);
                vector<string> b{a};
                temp = multiply(temp, b);
            } else if (*i == ',') {
                res.insert(res.end(), temp.begin(), temp.end());
                temp.clear();
            } else if (*i == '{') {
                int num = 1;
                auto j = i + 1;
                while (j != end && num > 0) {
                    if (*j == '{') {
                        ++num;
                    } else if (*j == '}') {
                        --num;
                    }
                    ++j;
                }
                auto brace = parseBrace(i + 1, j - 1);
                temp = multiply(temp, brace);
                i = j - 1;
            }
        }
        res.insert(res.end(), temp.begin(), temp.end());
        sort(res.begin(), res.end());
        return {res.begin(), unique(res.begin(), res.end())};
    }

    vector<string> braceExpansionII(string &expression) {
        return parseBrace(expression.begin(), expression.end());
    }

    bool minEatingSpeedHelper(vector<int> &piles, int h, long long k) {
        int all_hour = 0;
        for (auto i: piles) {
            all_hour += (i + k - 1) / k;
        }
        return all_hour <= h;
    }

    int minEatingSpeed(vector<int> &piles, int h) {
        if (piles.size() == 1) {
            return (piles[0] + h - 1) / h;
        }
        long long min_k = accumulate(begin(piles), end(piles), (long long) 0) / h;
        long long max_k = 1000000000;
        long long mid = (min_k + max_k) / 2;
        while (min_k < max_k) {
            if (minEatingSpeedHelper(piles, h, mid)) {
                max_k = mid;
            } else {
                min_k = mid + 1;
            }
            mid = (min_k + max_k) / 2;
        }
        return max_k;
    }

    int mostWordsFound(vector<string> &sentences) {
        int res = 0;
        int num;
        for (auto &i: sentences) {
            num = 1;
            for (auto &j: i) {
                if (j == ' ') {
                    ++num;
                }
            }
            if (num > res) {
                res = num;
            }
        }
        return res;
    }

    vector<string> findItinerary(vector<vector<string>> &tickets) {
        vector<string> res;
        vector<int> path;
        unordered_map<int, priority_queue<int, vector<int>, greater<>>> adj;

        function<int(string)> stringToInt = [](string input) -> int {
            return (((input[0] << 8) + input[1]) << 8) + input[2];
        };

        function<string(int)> intToString = [](int input) -> string {
            return {(char) (input >> 16), (char) ((input >> 8) % (1 << 8)), (char) (input % (1 << 8))};
        };

        function<void(int)> dfs = [&](int curr) {
            auto &adj_nodes = adj[curr];
            while (!adj_nodes.empty()) {
                auto temp = adj_nodes.top();
                adj_nodes.pop();
                dfs(temp);
            }
            path.emplace_back(curr);
        };

        for (auto &i: tickets)
            adj[stringToInt(i[0])].emplace(stringToInt(i[1]));

        dfs(stringToInt("JFK"));

        std::for_each(path.rbegin(), path.rend(), [&](const auto &item) {
            res.push_back(intToString(item));
        });
        return res;
    }
};

int main() {
    Solution s;
    string b = "AKA";
    cout << s.findItineraryIntToString(s.findItineraryStringToInt(b)) << endl;


}