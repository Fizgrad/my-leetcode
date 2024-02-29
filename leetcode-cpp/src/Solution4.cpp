//
// Created by david on 2024/2/14.
//
#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <deque>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right)
        : val(x), left(left), right(right) {}
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
            if (height[i] < height[j])
                ++i;
            else
                --j;
            res =
                max(res, static_cast<int>(min(height[i], height[j]) * (j - i)));
        }
        return res;
    }

    int findLeastNumOfUniqueInts(vector<int> &arr, int k) {
        unordered_map<int, int> freq;
        for (auto i : arr)
            ++freq[i];
        priority_queue<pair<int, int>, vector<pair<int, int>>,
                       greater<pair<int, int>>>
            pq;
        for (auto &k : freq) {
            pq.emplace(k.second, k.first);
        }
        while (k > 0) {
            if (k >= pq.top().first) {
                k -= pq.top().first;
                pq.pop();
            } else
                return pq.size();
        }
        return pq.size();
    }

    long long largestPerimeter(vector<int> &nums) {
        std::sort(nums.begin(), nums.end());
        if (nums.size() < 3)
            return -1;
        int n = nums.size();
        long long res = std::accumulate(nums.begin(), nums.end(),
                                        static_cast<long long>(0));
        int index = n - 1;
        while (index >= 1) {
            if (nums[index] >= res - nums[index]) {
                res -= nums[index--];
            } else
                return res;
        }
        return -1;
    }

    int furthestBuilding(vector<int> &heights, int bricks, int ladders) {
        priority_queue<int> pq;
        int origin_bricks = bricks;
        for (int i = 1; i < heights.size(); ++i) {
            int need = heights[i] - heights[i - 1];
            if (need <= 0)
                continue;
            if (bricks >= need) {
                bricks -= need;
                pq.emplace(need);
            } else if (origin_bricks < need) {
                if (ladders > 0)
                    --ladders;
                else
                    return i - 1;
            } else {
                bricks -= need;
                pq.emplace(need);
                while (bricks < 0 && ladders > 0 && !pq.empty()) {
                    --ladders;
                    bricks += pq.top();
                    pq.pop();
                }
                if (bricks < 0)
                    return i - 1;
            }
        }
        return heights.size() - 1;
    }

    int mostBooked(int n, vector<vector<int>> &meetings) {
        vector<int> rooms(n, 0);
        set<int> s;
        priority_queue<pair<int64_t, int64_t>, vector<pair<int64_t, int64_t>>,
                       greater<pair<int64_t, int64_t>>>
            q;
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

    int missingNumber(vector<int> &nums) {
        int res = 0;
        for (int i = 0; i < nums.size(); ++i) {
            res ^= nums[i];
            res ^= (i + 1);
        }
        return res;
    }

    int rangeBitwiseAnd(int left, int right) {
        int and_sum = left & right;
        int res = 0;
        int index = 30;
        while (index >= 0) {
            if ((right & (1 << index)) == (left & (1 << index))) {
                res |= (and_sum & (1 << index));
                --index;
            } else
                return res;
        }
        return res;
    }

    bool canTraverseAllPairs(vector<int> &nums) {
        auto primeFactors = [](int n) -> unordered_set<int> {
            unordered_set<int> res;
            while (n % 2 == 0) {
                res.insert(2);
                n = n / 2;
            }
            for (int i = 3; i <= sqrt(n); i = i + 2) {
                while (n % i == 0) {
                    res.insert(i);
                    n = n / i;
                }
            }
            if (n > 2)
                res.insert(n);
            return res;
        };
        unordered_map<int, unordered_set<int>> prime2index;
        unordered_map<int, unordered_set<int>> index2prime;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            index2prime[i] = primeFactors(nums[i]);
            for (auto k : index2prime[i]) {
                prime2index[k].insert(i);
            }
        }
        unordered_map<int, bool> isVisited;
        vector<bool> isConnected(n, 0);
        auto dfs = [&](auto &&dfs, int index) -> void {
            for (auto prime : index2prime[index]) {
                if (isVisited[prime])
                    continue;
                isVisited[prime] = true;
                for (auto i : prime2index[prime]) {
                    if (i == index || isConnected[i])
                        continue;
                    isConnected[i] = true;
                    dfs(dfs, i);
                }
            }
        };
        isConnected[0] = true;
        dfs(dfs, 0);
        return std::all_of(begin(isConnected), end(isConnected),
                           [](auto item) -> bool { return item == true; });
    }

    vector<int> findAllPeople(int n, vector<vector<int>> &meetings,
                              int firstPerson) {
        sort(meetings.begin(), meetings.end(),
             [](const auto &l, const auto &r) { return l[2] < r[2]; });

        vector<int> res;
        vector<int> parents(n + 1, 0);
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
            parents[max(pa, pb)] = min(pa, pb);
        };

        uf_union(0, firstPerson);
        int i = 0;
        vector<int> temp;
        while (i < meetings.size()) {
            int currentTime = meetings[i][2];
            temp.clear();
            while (i < meetings.size() && meetings[i][2] == currentTime) {
                int g1 = uf_find(meetings[i][0]);
                int g2 = uf_find(meetings[i][1]);
                uf_union(g1, g2);
                temp.push_back(meetings[i][0]);
                temp.push_back(meetings[i][1]);
                ++i;
            }
            for (int j = 0; j < temp.size(); ++j) {
                if (uf_find(temp[j]) != uf_find(0)) {
                    parents[temp[j]] = temp[j];
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (uf_find(i) == uf_find(0)) {
                res.emplace_back(i);
            }
        }
        return {res.begin(), res.end()};
    }

    bool isSameTree(TreeNode *p, TreeNode *q) {
        auto isSameTreeDFS = [](auto &&isSameTreeDFS, TreeNode *p,
                                TreeNode *q) -> bool {
            if (p == nullptr && q == nullptr) {
                return true;
            }
            if (p == nullptr || q == nullptr) {
                return false;
            }
            return p->val == q->val &&
                   isSameTreeDFS(isSameTreeDFS, p->left, q->left) &&
                   isSameTreeDFS(isSameTreeDFS, p->right, q->right);
        };
        return isSameTreeDFS(isSameTreeDFS, p, q);
    }

    int diameterOfBinaryTree(TreeNode *root) {
        int res = 0;
        auto dfs = [&](auto &&dfs, TreeNode *node) -> int {
            if (node == nullptr)
                return 0;
            int left = dfs(dfs, node->left);
            int right = dfs(dfs, node->right);
            res = max(res, left + right);
            return max(right, left) + 1;
        };
        dfs(dfs, root);
        return res;
    }

    int findBottomLeftValue(TreeNode *root) {
        int res = root->val;
        int level = 0;
        auto dfs = [&](auto &&dfs, TreeNode *node, int depth) {
            if (node == nullptr)
                return;
            if (depth > level) {
                level = depth;
                res = node->val;
            }
            dfs(dfs, node->left, depth + 1);
            dfs(dfs, node->right, depth + 1);
        };
        dfs(dfs, root, 0);
        return res;
    }

    int lengthOfLongestSubstring(const string &s) {
        int n = s.size();
        if (n == 0)
            return 0;
        int left = 0;
        int right = 0;
        int res = 1;
        unordered_map<char, int> noCollisionIndices;
        noCollisionIndices[*s.begin()] = 1;
        for (right = 0; right < n; ++right) {
            res = max(right - left + 1, res);
            char c = s[right + 1];
            if (noCollisionIndices.find(c) != end(noCollisionIndices)) {
                left = max(left, noCollisionIndices[c]);
            }
            noCollisionIndices[c] = right + 1 + 1;
        }
        return res;
    }

    double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
        int n = nums1.size();
        int m = nums2.size();
        int i = 0;
        int j = 0;
        vector<int> nums(n + m, 0);
        int k = 0;
        while (i < n && j < m) {
            if (nums1[i] > nums2[j]) {
                nums[k++] = nums2[j++];
            } else {
                nums[k++] = nums1[i++];
            }
        }
        while (i < n)
            nums[k++] = nums1[i++];
        while (j < m)
            nums[k++] = nums2[j++];
        bool even = k & 1;
        if (even) {
            return nums[(k - 1) >> 1];
        } else {
            return (static_cast<double>(nums[k >> 1]) + nums[(k >> 1) - 1]) / 2;
        }
    }

    int reverse(int x) {
        bool negtive = (x < 0);
        if (x == INT32_MIN)
            return 0;
        x = negtive ? -x : x;
        string xString = to_string(x);
        stringstream sstream;
        sstream << string(xString.rbegin(), xString.rend());
        sstream >> x;
        if (x == INT32_MAX || x == INT32_MIN)
            x = 0;
        x = negtive ? -x : x;
        return x;
    }

    void setZeroes(vector<vector<int>> &matrix) {
        int n = matrix.size();
        int m = matrix.begin()->size();
        vector<bool> cols(m, 0);
        vector<bool> rows(n, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (matrix[i][j] == 0) {
                    cols[j] = true;
                    rows[i] = true;
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (cols[j] || rows[i])
                    matrix[i][j] = 0;
            }
        }
    }

    bool isEvenOddTree(TreeNode *root) {
        int level = 0;
        vector<TreeNode *> next;
        vector<TreeNode *> nodes;
        nodes.emplace_back(root);
        while (!nodes.empty()) {
            if (level & 1) {
                for (int i = 1; i < nodes.size(); ++i) {
                    if (nodes[i - 1]->val <= nodes[i]->val) {
                        return false;
                    }
                }
            } else {
                for (int i = 1; i < nodes.size(); ++i) {
                    if (nodes[i - 1]->val >= nodes[i]->val) {
                        return false;
                    }
                }
            }
            for (auto i : nodes) {
                if ((i->val & 1) == (level & 1))
                    return false;
                if (i->left)
                    next.emplace_back(i->left);
                if (i->right)
                    next.emplace_back(i->right);
            }
            nodes.clear();
            nodes.swap(next);
            ++level;
        }
        return true;
    }
};

int main() { return 0; }