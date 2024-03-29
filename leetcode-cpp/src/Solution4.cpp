//
// Created by david on 2024/2/14.
//
#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <type_traits>
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
};

int main() { return 0; }