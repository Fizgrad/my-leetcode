//
// Created by Fitzgerald on 2/19/23.
//
#include<iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <array>
#include <deque>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

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

    int strStr2(string haystack, string needle) {
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
    int findKthPositive(vector<int>& arr, int k) {
        int index = 0;
        int i ;
        for(i = 1;i<=arr[arr.size()-1];++i){
            if(i == arr[index]){
                ++index;
            }else {
                --k;
                if (k == 0){
                    return i;
                }
            }
        }
        while(--k){
            ++i;
        }
        return i;
    }
};

int main() {
    Solution s;
    vector<int> a = {1, 3, 5, 2, 7, 5};
    cout << s.countSubarrays(a, 1, 1);
}