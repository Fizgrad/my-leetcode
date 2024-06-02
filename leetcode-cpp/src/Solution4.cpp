//
// Created by david on 2024/2/14.
//
#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
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
    ListNode(int x, ListNode *next) : val(x), next(next) {}
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
        for (auto i: arr)
            ++freq[i];
        priority_queue<pair<int, int>, vector<pair<int, int>>,
                       greater<pair<int, int>>>
                pq;
        for (auto &k: freq) {
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
            for (auto k: index2prime[i]) {
                prime2index[k].insert(i);
            }
        }
        unordered_map<int, bool> isVisited;
        vector<bool> isConnected(n, 0);
        auto dfs = [&](auto &&dfs, int index) -> void {
            for (auto prime: index2prime[index]) {
                if (isVisited[prime])
                    continue;
                isVisited[prime] = true;
                for (auto i: prime2index[prime]) {
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
            for (auto i: nodes) {
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

    string maximumOddBinaryNumber(const string &s) {
        int num_one = 0;
        for (auto i: s) num_one += (i == '1');
        return string(num_one - 1, '1') + string(s.size() - num_one, '0') + "1";
    }

    vector<int> sortedSquares(vector<int> &nums) {
        for (auto &i: nums) i = i * i;
        std::sort(nums.begin(), nums.end());
        return nums;
    }

    ListNode *removeNthFromEnd(ListNode *head, int n) {
        ListNode *removedNext = nullptr;
        auto f = [&](auto &&f, ListNode *node) -> int {
            if (node == nullptr) return 0;
            int rIndex = 1 + f(f, node->next);
            if (rIndex == n) removedNext = node->next;
            if (rIndex == n + 1) node->next = removedNext;
            return rIndex;
        };
        ListNode *virtualNode = new ListNode(0, head);
        f(f, virtualNode);
        return virtualNode->next;
    }

    int bagOfTokensScore(vector<int> &tokens, int power) {
        sort(begin(tokens), end(tokens));
        int score = 0;
        int res = 0;
        int i = 0;
        int j = tokens.size() - 1;
        while (j >= i) {
            while (i < tokens.size() && power >= tokens[i]) {
                power -= tokens[i++];
                res = max(res, ++score);
            }
            if (j < i || score == 0) return res;
            power += tokens[j--];
            --score;
        }
        return res;
    }

    int minimumLength(const string &s) {
        int i = 0;
        int j = s.size() - 1;
        while (i <= j) {
            if (i == j) return 1;
            auto c = s[i];
            if (s[j] == c) {
                while (j > i && s[j] == c) {
                    --j;
                }
                while (i < j && s[i] == c) {
                    ++i;
                }
                if (i == j) return (s[i] == c) ? 0 : 1;
            } else
                return j - i + 1;
        }
        return j - i + 1;
    }

    bool hasCycle(ListNode *head) {
        ListNode *slow = head;
        ListNode *fast = head;
        if (head == nullptr) return false;
        do {
            if (slow->next != nullptr) {
                slow = slow->next;
            } else {
                return false;
            }
            if (fast->next != nullptr) {
                fast = fast->next;
            } else {
                return false;
            }
            if (fast->next != nullptr) {
                fast = fast->next;
            } else {
                return false;
            }
        } while (fast != slow);
        return true;
    }

    int maxFrequencyElements(vector<int> &nums) {
        int max_freq = 1;
        unordered_map<int, int> times;
        int res = 0;
        for (auto i: nums) {
            ++times[i];
            if (times[i] > max_freq) {
                res = times[i];
                max_freq = times[i];
            } else if (times[i] == max_freq) {
                res += times[i];
            }
        }
        return res;
    }

    ListNode *middleNode(ListNode *head) {
        auto fast = head;
        auto slow = head;
        while (fast->next) {
            if (slow->next != nullptr) {
                slow = slow->next;
            } else
                return slow;
            if (fast->next != nullptr) {
                fast = fast->next;
            } else
                return slow;
            if (fast->next != nullptr) {
                fast = fast->next;
            } else
                return slow;
        }
        return slow;
    }

    int getCommon(vector<int> &nums1, vector<int> &nums2) {
        int i = 0;
        int j = 0;
        while (i < nums1.size() && j < nums2.size()) {
            if (nums1[i] == nums2[j]) return nums1[i];
            if (nums1[i] > nums2[j]) ++j;
            else
                ++i;
        }
        return -1;
    }

    vector<int> intersection(vector<int> &nums1, vector<int> &nums2) {
        unordered_set<int> set1;
        unordered_set<int> set2;
        for (auto i: nums1) set1.insert(i);
        for (auto i: nums2) set2.insert(i);
        vector<int> res;
        for (auto i: set1)
            if (set2.find(i) != set2.end()) res.emplace_back(i);
        return res;
    }

    string customSortString(string order, string s) {
        int i = 0;
        vector<int> weights('z' - 'a' + 1, 0);
        for (auto c: order) {
            weights[c - 'a'] = i++;
        }
        std::sort(s.begin(), s.end(), [&](const auto &a, const auto &b) -> bool {
            return weights[a - 'a'] < weights[b - 'a'];
        });
        return s;
    }

    ListNode *removeZeroSumSublists(ListNode *head) {
        int sum = 0;
        int index = 0;
        auto pt = head;
        unordered_map<int, ListNode *> index2node;
        unordered_map<ListNode *, int> node2sum;
        unordered_map<int, int> sum2index;
        auto virtual_node = new ListNode(0);
        virtual_node->next = head;
        index2node[index] = virtual_node;
        sum2index[0] = index;
        node2sum[virtual_node] = 0;
        while (pt != nullptr) {
            sum += pt->val;
            index += 1;
            index2node[index] = pt;
            node2sum[pt] = sum;
            if (sum2index.find(sum) == sum2index.end()) {
                sum2index[sum] = index;
            } else {
                auto prev_index = sum2index[sum];
                auto prev_node = index2node[prev_index];
                auto remove_node = prev_node->next;
                while (remove_node != nullptr && remove_node != pt->next) {
                    auto remove_sum = node2sum[remove_node];
                    if (index2node[sum2index[remove_sum]] == remove_node)
                        sum2index.erase(remove_sum);
                    remove_node = remove_node->next;
                }
                prev_node->next = pt->next;
                sum2index[sum] = prev_index;
            }
            pt = pt->next;
        }
        if (sum2index.find(0) != sum2index.end()) {
            auto prev_index = sum2index[sum];
            auto prev_node = index2node[prev_index];
            prev_node->next = nullptr;
        }
        return virtual_node->next;
    }

    int pivotInteger(int n) {
        int x2 = n * (1 + n) / 2;
        int x = sqrt(x2);
        return x * x == x2 ? x : -1;
    }

    int numSubarraysWithSum(vector<int> &nums, int goal) {
        vector<int> gaps;
        int res = 0;
        int last_index = -1;
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] == 1) {
                gaps.push_back(i - last_index - 1);
                last_index = i;
            }
        }
        gaps.push_back(nums.size() - last_index - 1);
        if (goal == 0) {
            for (int i = 0; i < gaps.size(); ++i) {
                res += (gaps[i] + 1) * gaps[i] / 2;
            }
        } else if (goal == 1) {
            for (int i = 0; i + 1 < gaps.size(); ++i) {
                res += (gaps[i] + 1) * (gaps[i + 1] + 1);
            }
        } else {
            for (int i = 0; i + goal < gaps.size(); ++i) {
                res += (gaps[i] + 1) * (gaps[i + goal] + 1);
            }
        }
        return res;
    }

    vector<int> productExceptSelf(vector<int> &nums) {
        int product = nums[0];
        vector<int> res(nums.size(), 1);
        for (int i = 1; i < nums.size(); ++i) {
            res[i] *= product;
            product *= nums[i];
        }
        product = nums[nums.size() - 1];
        for (int i = nums.size() - 2; i >= 0; --i) {
            res[i] *= product;
            product *= nums[i];
        }
        return res;
    }

    vector<vector<int>> insert(vector<vector<int>> &intervals, vector<int> &newInterval) {
        vector<vector<int>> res;
        bool flag = 0;
        for (int i = 0; i < intervals.size(); ++i) {
            if (intervals[i][1] < newInterval[0]) {
                res.emplace_back(intervals[i]);
                continue;
            }
            if (newInterval[1] < intervals[i][0]) {
                if (!flag) res.emplace_back(newInterval);
                res.emplace_back(intervals[i]);
                flag = 1;
                continue;
            }
            newInterval = {min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])};
        }
        if (!flag) res.emplace_back(newInterval);
        return res;
    }

    int findMaxLength(vector<int> &nums) {
        int sum = 0;
        unordered_map<int, int> indices;
        indices[0] = -1;
        int res = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] == 0) --sum;
            else
                ++sum;
            if (indices.find(sum) == indices.end())
                indices[sum] = i;
            else {
                res = max(res, i - indices[sum]);
            }
        }
        return res;
    }

    int findMinArrowShots(vector<vector<int>> &points) {
        std::sort(begin(points), end(points), [](const auto &a, const auto &b) {
            return a[0] < b[0];
        });
        int res = 1;
        int x = points[0][1];
        for (int i = 1; i < points.size(); ++i) {
            if (x < points[i][0]) {
                x = points[i][1];
                ++res;
                continue;
            }
            if (x > points[i][1]) {
                x = points[i][1];
            }
        }
        return res;
    }

    int leastInterval(vector<char> &tasks, int n) {
        int max_freq = 0;
        int size = tasks.size();
        int nums['Z' - 'A' + 1] = {0};
        for (auto c: tasks) {
            max_freq = max(max_freq, ++nums[c - 'A']);
        }
        int num_max_freq = 0;
        for (auto i: nums) num_max_freq += (i == max_freq);
        int time = (max_freq - 1) * (n + 1) + num_max_freq;
        return max(time, size);
    }

    ListNode *mergeInBetween(ListNode *list1, int a, int b, ListNode *list2) {
        ListNode *b_next;
        ListNode *a_prev;
        auto virtual_node = new ListNode(0);
        virtual_node->next = list1;
        b_next = list1->next;
        while (b--) {
            b_next = b_next->next;
        }
        a_prev = virtual_node;
        for (int i = 0; i < a; ++i) {
            a_prev = a_prev->next;
        }
        a_prev->next = list2;
        auto list2_end = list2;
        while (list2_end->next != nullptr) {
            list2_end = list2_end->next;
        }
        list2_end->next = b_next;
        return virtual_node->next;
    }

    ListNode *reverseList(ListNode *head) {
        if (head == nullptr || head->next == nullptr)
            return head;
        auto prev = head;
        auto pt = head->next;
        head->next = nullptr;
        while (pt != nullptr) {
            auto next_pt = pt->next;
            pt->next = prev;
            prev = pt;
            pt = next_pt;
        }
        return prev;
    }

    void reorderList(ListNode *head) {
        stack<ListNode *> s;
        auto pt = head;
        while (pt != nullptr) {
            s.push(pt);
            pt = pt->next;
        }
        pt = head;
        while (true) {
            if (s.top() == pt || pt->next == s.top()) {
                s.top()->next = nullptr;
                return;
            }
            auto next_pt = pt->next;
            pt->next = s.top();
            s.top()->next = next_pt;
            s.pop();
            pt = next_pt;
        }
    }

    bool isPalindrome(ListNode *head) {
        if (head->next == nullptr) return true;
        ListNode *pt = head;
        ListNode *fast = head;
        while (fast->next != nullptr) {
            fast = fast->next;
            if (fast->next != nullptr) {
                fast = fast->next;
                pt = pt->next;
            }
        }
        ListNode *prev_pt = nullptr;
        ListNode *next_pt;
        while (pt) {
            next_pt = pt->next;
            pt->next = prev_pt;
            prev_pt = pt;
            pt = next_pt;
        }
        pt = prev_pt;
        while (pt != nullptr && head != nullptr) {
            if (pt->val == head->val) {
                pt = pt->next;
                head = head->next;
            } else
                return false;
        }
        return true;
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

    vector<int> findDuplicates(vector<int> &nums) {
        vector<int> result;
        for (auto i: nums) {
            auto index = std::abs(i) - 1;
            if (nums[index] < 0)
                result.push_back(std::abs(i));
            else
                nums[index] = -nums[index];
        }
        return result;
    }

    int firstMissingPositive(vector<int> &nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (nums[i] <= 0) {
                nums[i] = n + 100;
            }
        }
        for (int i = 0; i < n; ++i) {
            int index = abs(nums[i]) - 1;
            if (index >= n)
                continue;
            if (nums[index] > 0)
                nums[index] = -nums[index];
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }

    int numSubarrayProductLessThanK(vector<int> &nums, int k) {
        int product = nums[0];
        int res = product < k ? 1 : 0;
        int i = 0;
        int j = 0;
        int n = nums.size();
        while (j + 1 < n) {
            ++j;
            product *= nums[j];
            while (product >= k && i < j) {
                product /= nums[i];
                ++i;
            }
            if (product < k) {
                res += j - i + 1;
            }
        }
        return res;
    }

    int maxSubarrayLength(vector<int> &nums, int k) {
        int n = nums.size();
        int i = 0;
        int j = 0;
        int res = 1;
        unordered_map<int, int> freq;
        freq[nums[0]]++;
        while (j + 1 < n) {
            ++j;
            ++freq[nums[j]];
            while (freq[nums[j]] > k && i < j) {
                --freq[nums[i]];
                ++i;
            }
            if (freq[nums[j]] <= k) {
                res = max(res, j - i + 1);
            }
        }
        return res;
    }

    long long countSubarrays(vector<int> &nums, int k) {
        int n = nums.size();
        int max_num = *std::max_element(begin(nums), end(nums));
        vector<int> indices;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == max_num) {
                indices.push_back(i);
            }
        }
        if (indices.size() < k) return 0;
        long long res = 0;
        for (int i = 0; i < indices.size(); ++i) {
            long long front = indices[i] + 1;
            long long behind = (i + k < indices.size()) ? indices[i + k] - indices[i + k - 1] : (i + k == indices.size() ? n - indices.back() : 0);
            res += front * behind;
        }
        return res;
    }

    // 刚好 n -> (最大为n - 最大为n-1)
    int subarraysWithKDistinct(vector<int> &nums, int k) {
        auto atMostKDistinct = [](const vector<int> &nums, int k) {
            unordered_map<int, int> count;
            int total = 0, i = 0, distinct = 0;
            for (int j = 0; j < nums.size(); ++j) {
                if (!count[nums[j]]++) {
                    distinct++;// 新增一个不同的整数
                }
                while (distinct > k) {
                    if (!--count[nums[i]]) {
                        distinct--;// 移除一个不同的整数
                    }
                    i++;
                }
                total += j - i + 1;// 加上以nums[j]结尾的、最多有k个不同整数的子数组数量
            }
            return total;
        };
        return atMostKDistinct(nums, k) - atMostKDistinct(nums, k - 1);
    }

    //  1   2    3    4    5   6   7   8
    // bad good good min good max min good  <- now
    // res += 4 - 1
    // similar to subarray which elements are all equal to K
    long long countSubarrays(vector<int> &nums, int minK, int maxK) {
        long long res = 0;
        int bad_idx = -1, left_idx = -1, right_idx = -1;
        for (int i = 0; i < nums.size(); ++i) {
            if (!(minK <= nums[i] && nums[i] <= maxK)) {
                bad_idx = i;
            }
            if (nums[i] == minK) {
                left_idx = i;
            }
            if (nums[i] == maxK) {
                right_idx = i;
            }
            res += max(0, min(left_idx, right_idx) - bad_idx);
        }
        return res;
    }

    int lengthOfLastWord(const string &s) {
        int res = 0;
        int len = 0;
        for (auto i: s) {
            if (i == ' ') {
                len = 0;
            } else {
                ++len;
                res = len;
            }
        }
        return res;
    }

    bool isIsomorphic(const string &s, const string &t) {
        unordered_map<char, char> s2t;
        unordered_set<char> mapped;
        int n = s.size();
        for (int i = 0; i < n; ++i) {
            int ss = s[i];
            int tt = t[i];
            auto iter = s2t.find(ss);
            if (iter == s2t.end()) {
                if (mapped.count(tt)) {
                    return false;
                }
                s2t[ss] = tt;
                mapped.insert(tt);
            } else {
                if (iter->second != tt) {
                    return false;
                }
            }
        }
        return true;
    }

    bool exist(vector<vector<char>> &board, const string &word) {
        int n = board.size();
        int m = board.begin()->size();
        vector<vector<bool>> is_visited(n, vector<bool>(m, false));
        auto dfs = [&](auto &&dfs, int index, int x, int y) -> bool {
            if (word[index] == board[x][y]) {
                if (index == word.size() - 1) {
                    return true;
                }
                is_visited[x][y] = true;
                constexpr int d[5] = {0, -1, 0, 1, 0};
                for (int k = 0; k < 4; ++k) {
                    int xx = x + d[k];
                    int yy = y + d[k + 1];
                    if (xx < 0 || xx >= n || yy < 0 || yy >= m || is_visited[xx][yy]) {
                        continue;
                    } else {
                        if (dfs(dfs, index + 1, xx, yy))
                            return true;
                    }
                }
                is_visited[x][y] = false;
            }
            return false;
        };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (dfs(dfs, 0, i, j))
                    return true;
            }
        }
        return false;
    }

    int maxDepth(const string &s) {
        int res = 0;
        int now = 0;
        for (auto c: s) {
            if (c == '(') res = max(res, ++now);
            else if (c == ')')
                --now;
        }
        return res;
    }

    string makeGood(const string &s) {
        string res;
        char prev = s[0];
        res.push_back(s[0]);
        for (int i = 1; i < s.size(); ++i) {
            if (prev - s[i] == 'A' - 'a' || prev - s[i] == 'a' - 'A') {
                res.pop_back();
                if (res.empty())
                    prev = 127;
                else
                    prev = res.back();
            } else {
                res.push_back((prev = s[i]));
            }
        }
        return res;
    }

    string minRemoveToMakeValid(const string &s) {
        auto minRemoveToMakeValid = [](auto &&minRemoveToMakeValid, decltype(s.begin()) begin, decltype(s.end()) end) -> string {
            string res;
            for (auto i = begin; i != end; ++i) {
                if (*i != '(' && *i != ')') {
                    res.push_back(*i);
                } else if (*i == '(') {
                    int count = 1;
                    auto left_iter = i + 1;
                    while (left_iter != end) {
                        if (*left_iter == '(')
                            ++count;
                        else if (*left_iter == ')') {
                            --count;
                            if (count == 0) break;
                        }
                        ++left_iter;
                    }
                    if (left_iter != end) {
                        res += '(';
                        res += minRemoveToMakeValid(minRemoveToMakeValid, i + 1, left_iter);
                        res += ')';
                        i = left_iter;
                    }
                }
            }
            return res;
        };
        return minRemoveToMakeValid(minRemoveToMakeValid, s.begin(), s.end());
    }

    bool checkValidString(const string &s) {
        int stars = 0;
        int lefts = 0;
        int rights = 0;
        for (auto i = s.begin(); i != s.end(); ++i) {
            auto c = *i;
            if (c == '*') {
                ++stars;
            } else if (c == '(') {
                ++lefts;
            } else if (c == ')') {
                if (lefts == 0) {
                    --stars;
                } else {
                    --lefts;
                }
                if (stars < 0) {
                    return false;
                }
            }
        }
        stars = 0;
        lefts = 0;
        for (auto i = s.rbegin(); i != s.rend(); ++i) {
            auto c = *i;
            if (c == '*') {
                ++stars;
            } else if (c == ')') {
                ++rights;
            } else if (c == '(') {
                if (rights == 0) {
                    --stars;
                } else {
                    --rights;
                }
                if (stars < 0) {
                    return false;
                }
            }
        }
        return true;
    }

    int countStudents(vector<int> &students, vector<int> &sandwiches) {
        int n = students.size();
        int ones = 0;
        for (auto i: students) {
            ones += (i == 1);
        }
        int zeros = n - ones;
        for (auto i: sandwiches) {
            if (i == 1) {
                if (ones > 0)
                    --ones;
                else
                    return zeros;
            } else {
                if (zeros > 0)
                    --zeros;
                else
                    return ones;
            }
        }
        return 0;
    }

    int timeRequiredToBuy(vector<int> &tickets, int k) {
        for (int i = 0; i < k; ++i) {
            tickets[i] = min(tickets[i], tickets[k]);
        }
        for (int i = k + 1; i < tickets.size(); ++i) {
            tickets[i] = min(tickets[i], tickets[k] - 1);
        }
        return accumulate(begin(tickets), end(tickets), 0);
    }

    int sumNumbers(TreeNode *root) {
        if (root == nullptr)
            return 0;
        int res = 0;
        auto dfs = [&](auto &&dfs, TreeNode *node, int prev) -> void {
            prev *= 10;
            prev += node->val;
            if (node->left || node->right) {
                if (node->left)
                    dfs(dfs, node->left, prev);
                if (node->right)
                    dfs(dfs, node->right, prev);
            } else {
                res += prev;
            }
        };
        dfs(dfs, root, 0);
        return res;
    }

    int openLock(vector<string> &deadends, const string &target) {
        unordered_set<string> deadends_set;
        for (auto &i: deadends) {
            deadends_set.insert(i);
        }
        if (deadends_set.count("0000")) { return -1; }
        queue<pair<string, int>> queue;
        queue.push({"0000", 0});
        unordered_set<string> visited;
        visited.insert("0000");

        while (!queue.empty()) {
            auto current = queue.front();
            queue.pop();
            string currentCombination = current.first;
            int moves = current.second;

            if (currentCombination == target) {
                return moves;
            }
            for (int i = 0; i < 4; i++) {
                for (int delta: {-1, 1}) {
                    int newDigit = (currentCombination[i] - '0' + delta + 10) % 10;
                    string newCombination = currentCombination;
                    newCombination[i] = '0' + newDigit;
                    if (!visited.count(newCombination) && !deadends_set.count(newCombination)) {
                        visited.insert(newCombination);
                        queue.push({newCombination, moves + 1});
                    }
                }
            }
        }
        return -1;
    }

    vector<int> findMinHeightTrees(int n, vector<vector<int>> &edges) {
        if (n == 1) return {0};
        std::vector<std::vector<int>> adjacency(n);
        std::vector<int> degree(n, 0);
        for (auto &edge: edges) {
            int u = edge[0], v = edge[1];
            adjacency[u].push_back(v);
            adjacency[v].push_back(u);
            degree[u]++;
            degree[v]++;
        }

        std::deque<int> leaves;
        for (int i = 0; i < n; ++i) {
            if (degree[i] == 1) leaves.push_back(i);
        }

        int remainingNodes = n;
        while (remainingNodes > 2) {
            int leavesCount = leaves.size();
            remainingNodes -= leavesCount;
            for (int i = 0; i < leavesCount; ++i) {
                int leaf = leaves.front();
                leaves.pop_front();
                for (int neighbor: adjacency[leaf]) {
                    if (--degree[neighbor] == 1) {
                        leaves.push_back(neighbor);
                    }
                }
            }
        }
        return {leaves.begin(), leaves.end()};
    }

    int tribonacci(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;
        int prev1 = 0;
        int prev2 = 1;
        int prev3 = 1;
        int res;
        for (int i = 3; i <= n; ++i) {
            res = prev1 + prev2 + prev3;
            prev1 = prev2;
            prev2 = prev3;
            prev3 = res;
        }
        return res;
    }

    int longestIdealString(const string &s, int k) {
        int n = s.size();
        int res = 1;
        vector<int> dp(26, 0);
        for (auto c: s) {
            int tmp = 1;
            for (int i = max(0, c - k - 'a'); i < min(26, c + k + 1 - 'a'); ++i) {
                tmp = max(tmp, dp[i] + 1);
            }
            res = max(res, dp[c - 'a'] = max(dp[c - 'a'], tmp));
        }
        return res;
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>> &edges) {
        std::vector<std::vector<int>> adjacency(n);
        for (auto &edge: edges) {
            int u = edge[0], v = edge[1];
            adjacency[u].push_back(v);
            adjacency[v].push_back(u);
        }
        vector<int> res(n, 0);
        vector<int> count(n, 1);
        vector<int> sum(n, 0);
        auto dfs = [&](auto &&f, int node, int prev) -> void {
            for (auto i: adjacency[node]) {
                if (i != prev) {
                    f(f, i, node);
                    count[node] += count[i];
                    sum[node] += sum[i] + count[i];
                }
            }
        };
        dfs(dfs, 0, -1);
        auto reroot = [&](auto &&reroot, int node, int prev) -> void {
            for (auto i: adjacency[node]) {
                if (i != prev) {
                    res[i] = res[node] + n - 2 * count[i];
                    reroot(reroot, i, node);
                }
            }
        };
        res[0] = sum[0];
        reroot(reroot, 0, -1);
        return res;
    }

    int findRotateSteps(const string &ring, const string &key) {
        int ring_size = ring.size();
        auto clockwise = [&](int curr, int new_pos) {
            if (new_pos >= curr) {
                return new_pos - curr;
            }
            return ring_size - (curr - new_pos);
        };
        auto anti_clockwise = [&](int curr, int new_pos) {
            if (curr >= new_pos) {
                return curr - new_pos;
            }
            return ring_size - (new_pos - curr);
        };

        auto to = [&](int pos, int new_pos) {
            return min(anti_clockwise(pos, new_pos), clockwise(pos, new_pos));
        };

        int n = key.size();

        vector<vector<int>> dp(n, vector<int>(ring_size, -1));

        unordered_map<char, vector<int>> indices;
        for (int i = 0; i < ring_size; i++) {
            indices[ring[i]].push_back(i);
        }

        auto dfs = [&](auto &&dfs, int now, int pos) {
            if (now >= n) return 0;
            if (dp[now][pos] != -1) {
                return dp[now][pos];
            }

            int steps = INT_MAX;
            int key_value = key[now];

            for (int i = 0; i < indices[key_value].size(); i++) {
                int new_pos = indices[key_value][i];
                int taken = dfs(dfs, now + 1, new_pos);
                steps = min(steps, 1 + to(pos, new_pos) + taken);
            }
            return dp[now][pos] = steps;
        };
        return dfs(dfs, 0, 0);
    }

    int minFallingPathSum(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.begin()->size();
        vector<vector<int>> dp(n, vector<int>(m, -1));
        auto f = [&](auto &&f, int row, int col) {
            if (row >= n) {
                return 0;
            }
            if (dp[row][col] != -1) {
                return dp[row][col];
            }
            int minimun = std::numeric_limits<int>::max();
            for (int i = 0; i < m; ++i) {
                if (i != col) {
                    minimun = min(minimun, f(f, row + 1, i));
                }
            }
            return dp[row][col] = grid[row][col] + minimun;
        };
        if (n == 1 && m == 1) {
            return grid[0][0];
        }
        int res = std::numeric_limits<int>::max();
        for (int i = 0; i < m; ++i) {
            res = min(res, f(f, 0, i));
        }
        return res;
    }

    int minOperations(vector<int> &nums, int k) {
        std::bitset<32> target(k);
        int xor_sum = 0;
        for (auto i: nums) {
            xor_sum ^= i;
        }
        std::bitset<32> from(xor_sum);
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            if (target.test(i) != from.test(i)) {
                ++res;
            }
        }
        return res;
    }

    long long wonderfulSubstrings(const string &word) {
        vector<long long> cnt(1024, 0);
        cnt[0] = 1;
        int curState = 0;
        long long res = 0;
        for (char c: word) {
            int idx = c - 'a';
            curState ^= 1 << idx;
            res += cnt[curState];// even state
            for (char odd = 'a'; odd <= 'j'; odd++) {
                int oddState = curState ^ (1 << (odd - 'a'));
                res += cnt[oddState];
            }
            cnt[curState]++;
        }
        return res;
    }

    string reversePrefix(const string &word, char ch) {
        auto fist = word.find_first_of(ch);
        if (fist == word.size()) {
            return word;
        } else {
            string res = word.substr(0, fist + 1);
            std::reverse(res.begin(), res.end());
            return res + word.substr(fist + 1);
        }
    }

    int findMaxK(vector<int> &nums) {
        int k = -1;
        unordered_set<int> nums_set(nums.begin(), nums.end());
        for (auto i: nums) {
            if (i > k) {
                if (nums_set.count(-i)) {
                    k = i;
                }
            }
        }
        return k;
    }

    int compareVersion(const string &version1, const string &version2) {
        int n1 = version1.size(), n2 = version2.size();
        int x1 = 0, x2 = 0;
        for (int i = 0, j = 0; i < n1 || j < n2; i++, j++) {
            while (i < n1 && version1[i] != '.') {
                x1 = 10 * x1 + (version1[i++] - '0');
            }
            while (j < n2 && version2[j] != '.') {
                x2 = 10 * x2 + (version2[j++] - '0');
            }
            if (x1 < x2) return -1;
            else if (x1 > x2)
                return 1;
            x1 = 0;
            x2 = 0;
        }
        return 0;
    }

    int numRescueBoats(vector<int> &people, int limit) {
        sort(begin(people), end(people));
        int n = people.size();
        int i = 0;
        int j = n - 1;
        int res = 0;
        while (i <= j) {
            if (i == j) {
                return ++res;
            }
            if (people[i] + people[j] <= limit) {
                ++i;
                ++res;
                --j;
            } else {
                res += 1;
                --j;
            }
        }
        return res;
    }

    void deleteNode(ListNode *node) {
        auto pt = node;
        auto next = node->next;
        auto prev = node;
        while (next != nullptr) {
            pt->val = next->val;
            prev = pt;
            pt = next;
            next = next->next;
        }
        prev->next = nullptr;
    }

    ListNode *removeNodes(ListNode *head) {
        vector<ListNode *> nodes;
        auto pt = head;
        while (pt) {
            while (nodes.size() && pt->val > nodes.back()->val) {
                nodes.pop_back();
            }
            nodes.push_back(pt);
            pt = pt->next;
        }
        for (int i = 1; i < nodes.size(); ++i) {
            nodes[i - 1]->next = nodes[i];
        }
        return nodes[0];
    }

    ListNode *doubleIt(ListNode *head) {
        auto f = [&](auto &&f, ListNode *node) -> bool {
            if (node == nullptr) {
                return false;
            }
            bool flag = f(f, node->next);
            int sum = flag + node->val * 2;
            node->val = sum % 10;
            if (sum >= 10) return true;
            else
                return false;
        };
        if (f(f, head)) {
            ListNode *res = new ListNode(1);
            res->next = head;
            return res;
        }
        return head;
    }

    vector<string> findRelativeRanks(vector<int> &score) {
        int n = score.size();
        vector<pair<int, int>> temp;
        vector<string> res(n);
        temp.reserve(n);
        for (int i = 0; i < n; ++i) {
            temp.emplace_back(score[i], i);
        }
        sort(begin(temp), end(temp));
        res[temp[n - 1].second] = "Gold Medal";
        if (n <= 1) {
            return res;
        }
        res[temp[n - 2].second] = "Silver Medal";
        if (n <= 2) {
            return res;
        }
        res[temp[n - 3].second] = "Bronze Medal";
        if (n <= 3) {
            return res;
        }
        for (int i = n - 4; i >= 0; --i) {
            res[temp[i].second] = to_string(n - i);
        }
        return res;
    }

    long long maximumHappinessSum(vector<int> &happiness, int k) {
        sort(happiness.begin(), happiness.end(), greater<int>());
        long long int res = 0;
        for (int i = 0; i < k; i++) {
            res += (max(0, happiness[i] - i));
        }
        return res;
    }

    vector<vector<int>> largestLocal(vector<vector<int>> &grid) {
        int n = grid.size();
        int dx[9] = {1, 1, 1, 0, 0, -1, -1, -1, 0};
        int dy[9] = {1, 0, -1, 1, -1, 1, -1, 0, 0};
        vector<vector<int>> res(n - 2, vector<int>(n - 2));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < 9; ++k) {
                    int xx = i + dx[k];
                    int yy = j + dy[k];
                    if (xx >= 1 && yy >= 1 && xx <= n - 2 && yy <= n - 2) {
                        res[xx - 1][yy - 1] = max(res[xx - 1][yy - 1], grid[i][j]);
                    }
                }
            }
        }
        return res;
    }

    int matrixScore(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.front().size();
        int res = 0;

        auto flip_row = [&](int x) {
            for (int i = 0; i < m; ++i) {
                grid[x][i] = 1 - grid[x][i];
            }
        };
        auto flip_col = [&](int y) {
            for (int i = 0; i < n; ++i) {
                grid[i][y] = 1 - grid[i][y];
            }
        };

        auto count_col = [&](int y) {
            int res = 0;
            for (int i = 0; i < n; ++i) {
                res += grid[i][y];
            }
            return res;
        };

        for (int i = 0; i < n; ++i) {
            if (grid[i][0] == 0) {
                flip_row(i);
            }
        }

        auto get_row = [&](int x) {
            int res = 0;
            for (int i = 0; i < m; ++i) {
                res = res * 2 + grid[x][i];
            }
            return res;
        };

        for (int i = 0; i < m; ++i) {
            if (2 * count_col(i) < n) {
                flip_col(i);
            }
        }

        for (int i = 0; i < n; ++i) {
            res += get_row(i);
        }
        return res;
    }

    int getMaximumGold(vector<vector<int>> &grid) {
        int res = 0;

        int n = grid.size();
        int m = grid.front().size();
        int ds[5] = {0, 1, 0, -1, 0};

        vector<vector<bool>> isVisited(n, vector<bool>(m, false));

        auto dfs = [&](auto &&dfs, int x, int y, int sum) {
            if (x >= n || y >= m || x < 0 || y < 0 || grid[x][y] == 0) return;
            sum += grid[x][y];
            res = max(res, sum);
            for (int k = 0; k < 4; ++k) {
                int xx = ds[k] + x;
                int yy = ds[k + 1] + y;
                if (xx >= n || yy >= m || xx < 0 || yy < 0 || grid[xx][yy] == 0) continue;
                if (isVisited[xx][yy] == false) {
                    isVisited[xx][yy] = true;
                    dfs(dfs, xx, yy, sum);
                    isVisited[xx][yy] = false;
                }
            }
        };

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j]) {
                    isVisited[i][j] = true;
                    dfs(dfs, i, j, 0);
                    isVisited[i][j] = false;
                }
            }
        }
        return res;
    }

    int maximumSafenessFactor(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.front().size();
        int ds[5] = {0, 1, 0, -1, 0};
        int res = 0;
        vector<pair<int, int>> next;
        vector<pair<int, int>> temp;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == 1) {
                    next.emplace_back(i, j);
                }
            }
        }
        int dis = 0;
        while (next.size()) {
            temp.clear();
            ++dis;
            while (next.size()) {
                auto [x, y] = next.back();
                next.pop_back();
                for (int k = 0; k < 4; ++k) {
                    int xx = x + ds[k];
                    int yy = y + ds[k + 1];
                    if (xx < 0 || yy < 0 || xx >= n || yy >= m) {
                        continue;
                    }
                    if (grid[xx][yy] != 0) {
                        continue;
                    } else {
                        temp.emplace_back(xx, yy);
                        grid[xx][yy] = -dis;
                    }
                }
            }
            temp.swap(next);
        }
        priority_queue<tuple<int, int, int>> pq;
        if (grid[0][0] == 1 || grid[n - 1][m - 1] == 1) return 0;
        pq.emplace(-grid[0][0], 0, 0);
        vector<vector<int>> visited(n, vector<int>(m, 0));
        while (pq.size()) {
            auto [safe, x, y] = pq.top();
            pq.pop();
            if (x == n - 1 && y == m - 1) {
                res = max(res, safe);
                return res;
            }
            if (safe <= res) {
                continue;
            }
            if (visited[x][y] >= safe) {
                continue;
            } else {
                visited[x][y] = safe;
            }
            for (int k = 0; k < 4; ++k) {
                int xx = x + ds[k];
                int yy = y + ds[k + 1];
                if (xx < 0 || yy < 0 || xx >= n || yy >= m) {
                    continue;
                }
                if (grid[xx][yy] == 1) {
                    int new_ans = min(safe, 0);
                    if (new_ans > res)
                        pq.emplace(new_ans, xx, yy);

                } else if (grid[xx][yy] < 0) {
                    int new_ans = min(safe, -grid[xx][yy]);
                    if (new_ans > res)
                        pq.emplace(new_ans, xx, yy);
                }
            }
        }
        return res;
    }

    bool evaluateTree(TreeNode *root) {
        auto &&dfs = [&](auto &&dfs, TreeNode *node) -> bool {
            if (node == nullptr) {
                return false;
            }
            if (node->left && node->right) {
                if (node->val == 2) {
                    return dfs(dfs, node->left) || dfs(dfs, node->right);
                }
                if (node->val == 3) {
                    return dfs(dfs, node->left) && dfs(dfs, node->right);
                }
            }
            return node->val;
        };
        return dfs(dfs, root);
    }

    TreeNode *removeLeafNodes(TreeNode *root, int target) {
        auto dfs = [&](auto &&dfs, TreeNode *node) -> bool {
            if (!node) {
                return false;
            }
            if (node->left) {
                if (dfs(dfs, node->left))
                    node->left = nullptr;
            }
            if (node->right) {
                if (dfs(dfs, node->right))
                    node->right = nullptr;
            }
            if (!(node->left || node->right)) {
                if (node->val == target) {
                    return true;
                }
            }
            return false;
        };
        if (dfs(dfs, root)) {
            return nullptr;
        }
        return root;
    }

    int distributeCoins(TreeNode *root) {
        int res = 0;
        auto dfs = [&](auto &&dfs, TreeNode *node) -> int {
            if (node == nullptr) return 0;
            int left = dfs(dfs, node->left);
            int right = dfs(dfs, node->right);
            int sum = node->val - 1 + left + right;
            res += std::abs(left) + std::abs(right);
            return sum;
        };
        dfs(dfs, root);
        return res;
    }

    long long maximumValueSum(vector<int> &nums, int k, vector<vector<int>> &edges) {
        long long sum = std::accumulate(begin(nums), end(nums), 0ll);
        long long max_neg_diff = std::numeric_limits<int>::min();
        long long min_pos_diff = std::numeric_limits<int>::max();
        long long diff_sum = 0;
        long long pos_diff_nums = 0;
        for (auto i: nums) {
            long long diff = (i ^ k) - i;
            if (diff >= 0) {
                diff_sum += diff;
                ++pos_diff_nums;
                min_pos_diff = min(min_pos_diff, diff);
            } else {
                max_neg_diff = max(max_neg_diff, diff);
            }
        }
        if (min_pos_diff == std::numeric_limits<int>::max())
            return sum;
        if (pos_diff_nums & 1) {
            if (max_neg_diff == std::numeric_limits<int>::min()) {
                return diff_sum + sum - min_pos_diff;
            }
            if (max_neg_diff + min_pos_diff > 0) {
                return diff_sum + sum + max_neg_diff;
            } else {
                return diff_sum + sum - min_pos_diff;
            }
        } else {
            return diff_sum + sum;
        }
    }

    int subsetXORSum(vector<int> &nums) {
        int res = 0;
        int n = nums.size();
        auto dfs = [&](auto &&dfs, int index, int sum) {
            if (index >= n) return;
            dfs(dfs, index + 1, sum);
            sum ^= nums[index];
            res += sum;
            dfs(dfs, index + 1, sum);
        };
        dfs(dfs, 0, 0);
        return res;
    }

    vector<vector<int>> subsets(vector<int> &nums) {
        vector<vector<int>> res;
        int n = nums.size();
        vector<int> temp;
        auto dfs = [&](auto &&dfs, int index) {
            if (index >= n) {
                return;
            }
            dfs(dfs, index + 1);
            temp.emplace_back(nums[index]);
            res.emplace_back(temp);
            dfs(dfs, index + 1);
            temp.pop_back();
        };
        res.emplace_back();
        dfs(dfs, 0);
        return res;
    }

    vector<vector<string>> partition(const string &s) {
        vector<vector<string>> res;
        int n = s.size();

        auto is_palindrome = [&](int from, int to) {
            while (from < to) {
                if (s[from] != s[to]) {
                    return false;
                }
                ++from;
                --to;
            }
            return true;
        };

        vector<string> temp;
        auto dfs = [&](auto &&dfs, int index) {
            if (index >= n) {
                res.push_back(temp);
                return;
            }
            for (int i = index; i < n; ++i) {
                if (is_palindrome(index, i)) {
                    temp.emplace_back(s.substr(index, i - index + 1));
                    dfs(dfs, i + 1);
                    temp.pop_back();
                }
            }
        };

        dfs(dfs, 0);
        return res;
    }

    int beautifulSubsets(vector<int> &nums, int k) {
        unordered_map<int, int> hashmap;
        int n = nums.size();
        auto dfs = [&](auto &&dfs, int index) {
            if (index == n) return 1;
            int num = nums[index];
            int taken = 0;
            if (!(hashmap[num + k] || hashmap[num - k])) {
                hashmap[nums[index]]++;
                taken = dfs(dfs, index + 1);
                hashmap[nums[index]]--;
            }
            int notTaken = dfs(dfs, index + 1);
            return taken + notTaken;
        };
        return dfs(dfs, 0) - 1;
    }

    int checkRecord(int n) {
        int res = 0;
        constexpr int MOD = 1e9 + 7;
        vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(2, vector<int>(3, -1)));
        // The parameter "day" represents the number of remaining days to choose P, L or A.
        auto f = [&](auto &&f, int day, int absent, int late) -> long long {
            if (day == 0) return 1;
            if (dp[day][absent][late] != -1) return dp[day][absent][late];
            long long ans = f(f, day - 1, absent, 0);
            if (late < 2)
                ans += f(f, day - 1, absent, late + 1);
            if (absent == 0)
                ans += f(f, day - 1, absent + 1, 0);
            return dp[day][absent][late] = ans % MOD;
        };
        // So here we need only f(f, n, 0, 0). At the beginning, "absent" and "late" must be both zero.
        return f(f, n, 0, 0) % MOD;
    }

    vector<string> wordBreak(const string &s, vector<string> &wordDict) {
        string temp;
        vector<string> res;
        Trie trie;
        for (auto &i: wordDict) {
            trie.add(i);
        }
        auto root = trie.root;
        auto f = [&](auto &&f, int index) {
            if (index >= s.size()) {
                res.emplace_back(temp);
                return;
            }
            auto pt = root;
            string subtemp = "";
            while (index < s.size()) {
                auto c = s[index];
                if (pt->next[c - 'a']) {
                    pt = pt->next[c - 'a'];
                    subtemp.push_back(c);
                    ++index;
                    if (pt->isEnd) {
                        int append_size = 0;
                        if (temp.size()) {
                            temp.append(" ");
                            ++append_size;
                        }
                        temp.append(subtemp);
                        append_size += subtemp.size();
                        f(f, index);
                        for (int i = 0; i < append_size; ++i) {
                            temp.pop_back();
                        }
                    }
                } else {
                    break;
                }
            }
        };
        f(f, 0);
        return res;
    }

    int maxScoreWords(vector<string> &words, vector<char> &letters, vector<int> &score) {
        int res = 0;
        vector<int> chars(26);
        vector<int> word_count(26, 0);
        for (auto i = 0; i < letters.size(); ++i) {
            ++chars[letters[i] - 'a'];
        }
        int score_sum = 0;
        auto f = [&](auto &&f, int index) {
            if (index >= words.size()) {
                res = max(res, score_sum);
                return;
            }
            f(f, index + 1);
            auto &word = words[index];
            std::fill_n(begin(word_count), 26, 0);
            for (auto i: word) {
                ++word_count[i - 'a'];
            }
            for (auto i = 0; i < 26; ++i) {
                if (word_count[i] > chars[i]) {
                    res = max(res, score_sum);
                    return;
                }
            }
            for (auto i: word) {
                score_sum += score[i - 'a'];
                --chars[i - 'a'];
            }
            f(f, index + 1);
            for (auto i: word) {
                score_sum -= score[i - 'a'];
                ++chars[i - 'a'];
            }
        };
        f(f, 0);
        return res;
    }

    int specialArray(vector<int> &nums) {
        std::sort(begin(nums), end(nums));
        int left = 0;
        int right = 100;
        while (left < right) {
            int mid = (left + right) >> 1;
            auto i = std::lower_bound(begin(nums), end(nums), mid);
            auto num = end(nums) - i;
            if (num == mid) {
                return mid;
            } else if (num > mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (end(nums) - std::lower_bound(begin(nums), end(nums), left) == left) return left;
        return -1;
    }

    int equalSubstring(const string &s, const string &t, int maxCost) {
        int n = s.size();
        vector<int> cost(n, 0);
        for (int i = 0; i < n; ++i) {
            cost[i] = std::abs(s[i] - t[i]);
        }
        int i = 0;
        int j = 0;
        int cost_sum = 0;
        int res = 0;
        do {
            cost_sum += cost[j];
            if (cost_sum <= maxCost) {
                res = max(res, j - i + 1);
            }
            while (cost_sum > maxCost) {
                cost_sum -= cost[i];
                ++i;
            }
            ++j;
        } while (j < n);
        return res;
    }

    int numSteps(string s) {
        int n = s.size();
        int j = n - 1;
        int res = 0;
        while (j > 0) {
            if (s[j] == '0') {
                --j;
                ++res;
                continue;
            } else {
                int k = 1;
                while (j - k >= 0 && s[j - k] == '1') {
                    ++k;
                }
                if (k > j) {
                    res += 1 + k;
                    j = j - k;
                } else {
                    s[j - k] = '1';
                    res += 1 + k;
                    j = j - k;
                }
            }
        }
        return res;
    }

    int countTriplets(vector<int> &arr) {
        int n = arr.size();
        int res = 0;
        vector<int> prefix_sums(n + 1, 0);
        int prefix = 0;
        for (int i = 0; i < n; ++i) {
            prefix = prefix_sums[i + 1] = (prefix ^ arr[i]);
        }
        for (int i = 0; i < n + 1; ++i) {
            for (int j = i + 2; j < n + 1; ++j) {
                for (int k = i + 1; k < j; ++k) {
                    int a = prefix_sums[k] ^ prefix_sums[i];
                    int b = prefix_sums[j] ^ prefix_sums[k];
                    if (a == b) {
                        ++res;
                    }
                }
            }
        }
        return res;
    }

    int scoreOfString(const string &s) {
        int res = 0;
        for (int i = 1; i < s.size(); ++i) {
            res += (std::abs(s[i - 1] - s[i]));
        }
        return res;
    }

    void reverseString(vector<char> &s) {
        int i = 0;
        int j = s.size() - 1;
        while (i < j) {
            auto temp = s[j];
            s[j] = s[i];
            s[i] = temp;
            ++i;
            --j;
        }
    }
};

int main() { return 0; }