//
// Created by david on 2024/2/14.
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
    int maxArea(vector<int> &height) {
        int res = min(height[0], height.back()) * (height.size() - 1);
        int i = 0;
        int j = height.size() - 1;
        while (i < j) {
            if (height[i] < height[j]) ++i; else --j;
            res = max(res, static_cast<int>(min(height[i], height[j]) * (j - i)));
        }
        return res;
    }

    int findLeastNumOfUniqueInts(vector<int> &arr, int k) {
        unordered_map<int, int> freq;
        for (auto i: arr) ++freq[i];
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        for (auto &k: freq) {
            pq.emplace(k.second, k.first);
        }
        while (k > 0) {
            if (k >= pq.top().first) {
                k -= pq.top().first;
                pq.pop();
            } else return pq.size();
        }
        return pq.size();
    }

    long long largestPerimeter(vector<int> &nums) {
        std::sort(nums.begin(), nums.end());
        if (nums.size() < 3) return -1;
        int n = nums.size();
        long long res = std::accumulate(nums.begin(), nums.end(), static_cast<long long>(0));
        int index = n - 1;
        while (index >= 1) {
            if (nums[index] >= res - nums[index]) {
                res -= nums[index--];
            } else return res;
        }
        return -1;
    }

    int furthestBuilding(vector<int> &heights, int bricks, int ladders) {
        priority_queue<int> pq;
        int origin_bricks = bricks;
        for (int i = 1; i < heights.size(); ++i) {
            int need = heights[i] - heights[i - 1];
            if (need <= 0)continue;
            if (bricks >= need) {
                bricks -= need;
                pq.emplace(need);
            } else if (origin_bricks < need) {
                if (ladders > 0) --ladders;
                else return i - 1;
            } else {
                bricks -= need;
                pq.emplace(need);
                while (bricks < 0 && ladders > 0 && !pq.empty()) {
                    --ladders;
                    bricks += pq.top();
                    pq.pop();
                }
                if (bricks < 0) return i - 1;
            }
        }
        return heights.size() - 1;
    }

    int mostBooked(int n, vector<vector<int>> &meetings) {
        vector<int> rooms(n, 0);
        set<int> s;
        priority_queue<pair<int64_t, int64_t>, vector<pair<int64_t, int64_t>>, greater<pair<int64_t, int64_t>>> q;
        sort(meetings.begin(), meetings.end());
        int m = meetings.size();
        for (int i = 0; i < n; i++) {
            s.insert(i);
        }
        for (int i = 0; i < m; i++) {
            int64_t start = meetings[i][0];
            int64_t end = meetings[i][1];
            while (q.size() > 0 && q.top().first <= start) {
                int room = q.top().second;
                q.pop();
                s.insert(room);
            }
            if (s.size() == 0) {
                pair<int64_t, int64_t> p = q.top();
                q.pop();
                int64_t dif = end - start;
                start = p.first;
                end = start + dif;
                s.insert(p.second);
            }
            auto it = s.begin();
            rooms[*it]++;
            q.push({end, *it});
            s.erase(*it);
        }
        int ans = 0;
        int64_t max_num = 0;
        for (int i = 0; i < n; i++) {
            if (max_num < rooms[i]) {
                max_num = rooms[i];
                ans = i;
            }
        }
        return ans;
    }

    bool isPowerOfTwo(int n) {
        return n != -2147483648 && __builtin_popcount(n) == 1;
    }

};

int main() {
    return 0;
}