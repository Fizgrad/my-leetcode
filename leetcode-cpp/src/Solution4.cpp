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
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <random>
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
    unordered_map<char, TrieNode *> next;
    bool isEnd = false;
};

struct Trie {
    TrieNode *root = new TrieNode();

    bool contains(const string &input) {
        auto temp = root;
        for (int i = 0; i < input.size(); ++i) {
            temp = temp->next[input[i]];
            if (temp == nullptr) {
                return false;
            }
        }
        return temp->isEnd;
    }

    void add(const string &input) {
        auto temp = root;
        for (int i = 0; i < input.size(); ++i) {
            if (temp->next[input[i]] == nullptr)
                temp->next[input[i]] = new TrieNode();
            temp = temp->next[input[i]];
        }
        temp->isEnd = true;
    }
};

struct PrefixTrieNode {
    unordered_map<char, PrefixTrieNode *> next;
    bool isEnd = false;
};

struct PrefixTrie {
    PrefixTrieNode *root = new PrefixTrieNode();

    bool contains(const string &input) {
        auto temp = root;
        for (int i = 0; i < input.size(); ++i) {
            temp = temp->next[input[i]];
            if (temp == nullptr) {
                return false;
            }
        }
        return temp->isEnd;
    }

    void add(const string &input) {
        auto temp = root;
        for (int i = 0; i < input.size(); ++i) {
            if (temp->next[input[i]] == nullptr)
                temp->next[input[i]] = new PrefixTrieNode();
            temp->isEnd = true;
            temp = temp->next[input[i]];
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

    int appendCharacters(const string &s, const string &t) {
        int res = 0;
        int j = 0;
        int i = 0;
        while (i < s.size() && j < t.size()) {
            if (s[i] == t[j]) {
                ++j;
            }
            ++i;
        }
        return t.size() - j;
    }

    int longestPalindrome(const string &s) {
        vector<bool> c(256, 0);
        int odd = 0;
        int res = 0;
        for (auto i: s) {
            if (c[i]) {
                --odd;
                res += 2;
            } else {
                ++odd;
            }
            c[i] = !c[i];
        }
        return res + (odd >= 1 ? 1 : 0);
    }

    vector<string> commonChars(vector<string> &words) {
        if (words.empty()) return {};
        multiset<char> word_set;
        multiset<char> res(words[0].begin(), words[0].end());
        multiset<char> temp;
        for (auto &word: words) {
            temp.clear();
            word_set.clear();
            word_set.insert(word.begin(), word.end());
            std::set_intersection(res.begin(), res.end(), word_set.begin(), word_set.end(), std::inserter(temp, temp.begin()));
            res.swap(temp);
        }
        return std::accumulate(res.begin(), res.end(), vector<string>(), [](vector<string> a, char b) {
            a.emplace_back(1, b);
            return std::move(a);
        });
    }

    bool isNStraightHand(vector<int> &hand, int groupSize) {
        int n = hand.size();
        if (n % groupSize) {
            return false;
        }
        unordered_map<int, int> nums;
        for (auto i: hand) {
            ++nums[i];
        }
        vector<int> keys;
        for (auto &i: nums) {
            keys.push_back(i.first);
        }
        std::sort(keys.begin(), keys.end());
        for (int i = 0; i < keys.size(); ++i) {
            int key = keys[i];
            if (nums[key] == 0) {
                continue;
            } else if (nums[key] < 0) {
                return false;
            } else {
                while (nums[key])
                    for (int k = 0; k < groupSize; ++k) {
                        --nums[key + k];
                        if (nums[key + k] < 0) {
                            return false;
                        }
                    }
            };
        }
        return true;
    }

    string replaceWords(vector<string> &dictionary, const string &sentence) {
        string res;
        Trie dict_trie;
        for (auto &i: dictionary) {
            dict_trie.add(i);
        }

        auto transform = [&](const string &s) {
            auto trie_node = dict_trie.root;
            for (int i = 0; i < s.size(); ++i) {
                trie_node = trie_node->next[s[i] - 'a'];
                if (trie_node) {
                    if (trie_node->isEnd) {
                        return s.substr(0, i + 1);
                    }
                } else {
                    break;
                }
            }
            return s;
        };

        auto i = sentence.begin();
        while (i < sentence.end()) {
            auto j = std::find(i, sentence.end(), ' ');
            if (res.size()) {
                res.push_back(' ');
            }
            res.append(transform(string(i, j)));
            i = j + 1;
        }
        return res;
    }

    bool checkSubarraySum(vector<int> &nums, int k) {
        int n = nums.size();
        int prefix_sum = nums[0];
        unordered_map<int, int> remains;
        remains[prefix_sum % k] = 0;
        for (int i = 1; i < n; ++i) {
            prefix_sum = prefix_sum + nums[i];
            int remain = prefix_sum % k;
            if (remain == 0) {
                return true;
            }
            if (remains.count(remain)) {
                if (i - remains[remain] > 1) {
                    return true;
                }
            } else
                remains[remain] = i;
        }
        return false;
    }

    int subarraysDivByK(vector<int> &nums, int k) {
        vector<int> remains(k, 0);
        int res = 0;
        long long int sum = 0;
        remains[sum]++;
        for (auto i: nums) {
            sum = (sum + i + k * 10000) % k;
            res += remains[sum % k]++;
        }
        return res;
    }

    int heightChecker(vector<int> &heights) {
        vector<int> sorted(heights);
        std::sort(heights.begin(), heights.end());
        int res = 0;
        for (int i = 0; i < heights.size(); ++i) {
            res += (heights[i] != sorted[i]);
        }
        return res;
    }

    vector<int> relativeSortArray(vector<int> &arr1, vector<int> &arr2) {
        unordered_map<int, int> nums;
        for (auto i: arr1) {
            ++nums[i];
        }
        vector<int> res;
        for (auto i: arr2) {
            while (nums[i]--) {
                res.push_back(i);
            }
        }
        int n = res.size();
        for (auto &i: nums) {
            while (i.second > 0) {
                res.push_back(i.first);
                --i.second;
            }
        }
        std::sort(res.begin() + n, res.end());
        return res;
    }

    void sortColors(vector<int> &nums) {
        int low = 0;
        int mid = 0;
        int high = nums.size() - 1;
        while (mid <= high) {
            if (nums[mid] == 0) {
                swap(nums[low++], nums[mid++]);
            } else if (nums[mid] == 1) {
                mid++;
            } else {
                swap(nums[mid], nums[high--]);
            }
        }
    }

    int minMovesToSeat(vector<int> &seats, vector<int> &students) {
        int n = seats.size();
        int res = 0;
        auto f = [&](auto &a) {
            std::sort(a.begin(), a.end());
        };
        f(seats);
        f(students);
        for (int i = 0; i < n; ++i) {
            res += std::abs(students[i] - seats[i]);
        }
        return res;
    }

    int minIncrementForUnique(vector<int> &nums) {
        std::sort(nums.begin(), nums.end());
        int res = 0;
        int n = nums.size();
        int prev = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] <= prev) {
                res += (prev + 1) - nums[i];
            }
            prev = max(prev + 1, nums[i]);
        }
        return res;
    }

    int minPatches(vector<int> &nums, int n) {
        int res = 0;
        int i = 0;
        long long int miss = 1;
        while (miss <= n) {
            if (i < nums.size() && nums[i] <= miss) {
                miss += nums[i];
                ++i;
            } else {
                miss += miss;
                ++res;
            }
        }
        return res;
    }

    int findMaximizedCapital(int k, int w, vector<int> &profits, vector<int> &capital) {
        int n = profits.size();
        vector<int> sorted(n, 0);
        for (int i = 0; i < n; ++i) {
            sorted[i] = i;
        }

        auto cmp = [&](int a, int b) {
            return capital[a] < capital[b];
        };

        std::sort(sorted.begin(), sorted.end(), cmp);

        int index = 0;
        priority_queue<int> profits_sorted;

        while (k--) {
            while (index < n && capital[sorted[index]] <= w) {
                profits_sorted.push(profits[sorted[index]]);
                ++index;
            }
            if (profits_sorted.size()) {
                w += profits_sorted.top();
                profits_sorted.pop();
            } else {
                return w;
            }
        }
        return w;
    }

    bool judgeSquareSum(int c) {
        long long int b = sqrt(c);
        long long int a = 0;
        while (a <= b) {
            long long int sum = a * a + b * b;
            if (sum < c) {
                ++a;
                continue;
            } else if (sum == c) {
                return true;
            } else {
                --b;
                continue;
            }
        }
        return false;
    }

    int maxProfitAssignment(vector<int> &difficulty, vector<int> &profit, vector<int> &worker) {
        int n = difficulty.size();
        int m = worker.size();

        vector<int> worker_sorted(m);

        std::function<bool(int, int)> cmp = [&](int a, int b) {
            return profit[a] < profit[b];
        };

        priority_queue<int, vector<int>, decltype(cmp)> available(cmp);

        for (int i = 0; i < m; ++i) {
            worker_sorted[i] = i;
        }

        for (int i = 0; i < n; ++i) {
            available.push(i);
        }

        std::sort(worker_sorted.begin(), worker_sorted.end(), [&](int a, int b) {
            return worker[a] > worker[b];
        });

        int res = 0;
        for (int i = 0; i < m; ++i) {
            int difficulty_now = worker[worker_sorted[i]];
            while (available.size() && difficulty[available.top()] > difficulty_now) {
                available.pop();
            }
            if (available.size()) {
                res += profit[available.top()];
            } else {
                return res;
            }
        }
        return res;
    }

    int minDays(vector<int> &bloomDay, int m, int k) {
        int n = bloomDay.size();
        int max_day = *std::max_element(bloomDay.begin(), bloomDay.end());
        if (static_cast<long long int>(m) * k > n) {
            return -1;
        }
        if (m * k == n) {
            return max_day;
        }
        int min_day = 1;
        auto f = [&](int day) {
            int bouquets = 0;
            int prev = 0;
            for (int i = 0; i < n; ++i) {
                if (bloomDay[i] <= day) {
                    ++prev;
                    if (prev == k) {
                        ++bouquets;
                        prev = 0;
                    }
                } else {
                    prev = 0;
                }
            }
            return bouquets >= m;
        };
        int res = -1;
        while (true) {
            if (max_day < min_day) {
                break;
            }
            int day = (max_day + min_day) >> 1;
            if (f(day)) {
                res = day;
                max_day = day - 1;
            } else {
                min_day = day + 1;
            }
        }
        return res;
    }

    int maxDistance(vector<int> &position, int m) {
        int min_dis = 0;
        int max_dis = INT_MAX;
        int res = 0;
        std::sort(position.begin(), position.end());
        auto f = [&](int dis) {
            int prev = 0;
            int remaining = m - 1;
            int index = 1;
            while (index < position.size()) {
                if (position[index] >= position[prev] + dis) {
                    --remaining;
                    prev = index;
                }
                ++index;
            }
            return remaining <= 0;
        };
        while (true) {
            if (min_dis > max_dis) {
                break;
            }
            int mid = (min_dis + max_dis) >> 1;
            if (f(mid)) {
                res = mid;
                min_dis = mid + 1;
            } else {
                max_dis = mid - 1;
            }
        }
        return res;
    }

    int maxSatisfied(vector<int> &customers, vector<int> &grumpy, int minutes) {
        int n = customers.size();
        int max_add = 0;
        int add_now = 0;
        for (int i = 0; i < minutes; ++i) {
            if (grumpy[i] == 1) {
                add_now += customers[i];
                max_add = max(max_add, add_now);
            }
        }
        for (int i = minutes; i < n; ++i) {
            if (grumpy[i - minutes] == 1)
                add_now -= customers[i - minutes];
            if (grumpy[i] == 1)
                add_now += customers[i];
            max_add = max(max_add, add_now);
        }
        int res = max_add;
        for (int i = 0; i < n; ++i) {
            if (grumpy[i] == 0) {
                res += customers[i];
            }
        }
        return res;
    }

    int numberOfSubarrays(vector<int> &nums, int k) {
        int n = nums.size();
        vector<int> cnt(n + 1, 0);
        cnt[0] = 1;
        int ans = 0, t = 0;
        for (int v: nums) {
            t += v & 1;
            if (t - k >= 0) {
                ans += cnt[t - k];
            }
            cnt[t]++;
        }
        return ans;
    }

    int longestSubarray(std::vector<int> &nums, int limit) {
        std::deque<int> maxq;
        std::deque<int> minq;
        int n = nums.size();
        int j = 0;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            while (!maxq.empty() && nums[i] > maxq.back())
                maxq.pop_back();
            maxq.push_back(nums[i]);
            while (!minq.empty() && nums[i] < minq.back()) {
                minq.pop_back();
            }
            minq.push_back(nums[i]);
            if (maxq.front() - minq.front() > limit) {
                if (nums[j] == maxq.front()) maxq.pop_front();
                if (nums[j] == minq.front()) minq.pop_front();
                j++;
            }
            ans = std::max(ans, i - j + 1);
        }
        return ans;
    }

    int minKBitFlips(vector<int> &nums, int k) {
        int n = nums.size();
        int res = 0;
        vector<bool> flip(n, 0);
        bool current = false;
        for (int i = 0; i < n; ++i) {
            if (i >= k) {
                current = current ^ flip[i - k];
            }
            if ((current ^ nums[i]) == 0) {
                if (i + k <= n) {
                    ++res;
                    flip[i] = 1;
                    current = !current;
                } else {
                    return -1;
                }
            }
        }
        return res;
    }

    int maxNumEdgesToRemove(int n, vector<vector<int>> &edges) {
        int num_of_edges = edges.size();
        vector<int> parent(n);
        vector<int> size(n);
        auto find = [&](int i) -> int {
            int res = parent[i];
            while (res != i) {
                i = res;
                res = parent[res];
            }
            return res;
        };
        auto uni = [&](int a, int b) -> void {
            int pa = find(a);
            int pb = find(b);
            if (pa == pb) {
                return;
            }
            if (size[pa] > size[pb]) {
                size[pa] += size[pb];
                parent[pb] = pa;
            } else {
                size[pb] += size[pa];
                parent[pa] = pb;
            }
        };
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            size[i] = 1;
        }
        int num = 0;
        for (auto &i: edges) {
            if (i[0] == 3) {
                if (find(i[1] - 1) != find(i[2] - 1)) {
                    uni(i[1] - 1, i[2] - 1);
                    ++num;
                }
            }
        }
        vector<int> parent_bak(parent);
        vector<int> size_bak(size);
        for (auto &i: edges) {
            if (i[0] == 1) {
                if (find(i[1] - 1) != find(i[2] - 1)) {
                    uni(i[1] - 1, i[2] - 1);
                    ++num;
                }
            }
        }
        int prev = find(0);
        for (int i = 1; i < n; ++i) {
            if (find(i) != prev) {
                return -1;
            }
        }
        parent = parent_bak;
        size = size_bak;
        for (auto &i: edges) {
            if (i[0] == 2) {
                if (find(i[1] - 1) != find(i[2] - 1)) {
                    uni(i[1] - 1, i[2] - 1);
                    ++num;
                }
            }
        }
        prev = find(0);
        for (int i = 1; i < n; ++i) {
            if (find(i) != prev) {
                return -1;
            }
        }
        return num_of_edges - num;
    }

    bool threeConsecutiveOdds(vector<int> &arr) {
        int prev = 0;
        for (auto i: arr) {
            if (i & 1) {
                ++prev;
                if (prev >= 3) {
                    return true;
                }
            } else {
                prev = 0;
            }
        }
        return false;
    }

    vector<int> intersect(vector<int> &nums1, vector<int> &nums2) {
        int times[1001] = {0};
        vector<int> res;
        for (auto i: nums1) {
            ++times[i];
        }
        for (auto i: nums2) {
            if (times[i]-- > 0) {
                res.push_back(i);
            }
        }
        return res;
    }

    TreeNode *bstToGst(TreeNode *root) {
        int addend = 0;
        auto dfs = [&](auto &&dfs, TreeNode *node) -> void {
            if (node == nullptr) {
                return;
            }
            dfs(dfs, node->right);
            node->val += addend;
            addend = node->val;
            dfs(dfs, node->left);
        };
        dfs(dfs, root);
        return root;
    }

    int findCenter(vector<vector<int>> &edges) {
        return edges[0][1] == edges[1][1] || edges[0][1] == edges[1][0] ? edges[0][1] : edges[0][0];
    }

    long long maximumImportance(int n, vector<vector<int>> &roads) {
        long long res = 0;
        long long value = 0;
        vector<int> degrees(n, 0);
        for (auto road: roads) {
            degrees[road[0]]++;
            degrees[road[1]]++;
        }
        std::sort(degrees.begin(), degrees.end());
        for (auto degree: degrees) res += degree * (++value);
        return res;
    }

    int minDifference(vector<int> &nums) {
        int n = nums.size();
        if (n <= 4) {
            return 0;
        }
        std::sort(nums.begin(), nums.end());
        return std::min(std::min(std::min(nums[n - 4] - nums[0], nums[n - 3] - nums[1]), nums[n - 2] - nums[2]), nums[n - 1] - nums[3]);
    }

    vector<int> nodesBetweenCriticalPoints(ListNode *head) {
        auto pt = head;
        int prev = head->val;
        int min_res = std::numeric_limits<int32_t>::max();
        int first_index = -1;
        int prev_index = -1;
        int max_res = -1;
        pt = pt->next;
        int index = 1;
        int prev_cmp = 0;// 1 greater, -1 smaller, 0 equal
        while (pt != nullptr) {
            int cmp = pt->val > prev ? 1 : (pt->val == prev ? 0 : -1);
            if (cmp != 0 && prev_cmp != 0 && cmp == -prev_cmp) {
                if (prev_index == -1) {
                    prev_index = index;
                } else {
                    min_res = std::min(min_res, index - prev_index);
                    prev_index = index;
                }
                if (first_index == -1) {
                    first_index = index;
                } else {
                    max_res = index - first_index;
                }
            }
            prev = pt->val;
            prev_cmp = cmp;
            pt = pt->next;
            ++index;
        }
        if (min_res == std::numeric_limits<int32_t>::max()) {
            return {-1, -1};
        }
        return {min_res, max_res};
    }

    double averageWaitingTime(vector<vector<int>> &customers) {
        int n = customers.size();
        double res = customers[0][1];
        int time = customers[0][0] + customers[0][1];
        for (int i = 1; i < n; ++i) {
            if (time < customers[i][0]) {
                time = customers[i][0];
            }
            time += customers[i][1];
            res += time - customers[i][0];
        }
        return res / n;
    }

    int passThePillow(int n, int time) {
        int chunks = time / (n - 1);
        return chunks % 2 == 0 ? (time % (n - 1) + 1) : (n - time % (n - 1));
    }

    int numWaterBottles(int numBottles, int numExchange) {
        return (numBottles - 1) / (numExchange - 1) * numExchange + (numBottles - 1) % (numExchange - 1) + 1;
    }

    int findTheWinner(int n, int k) {
        if (n == 1) return 1;
        return (findTheWinner(n - 1, k) + (k - 1)) % n + 1;
    }

    void flatten(TreeNode *root) {
        TreeNode *pt = nullptr;
        auto f = [&](auto &&f, TreeNode *node) -> void {
            if (node == nullptr) return;
            if (pt != nullptr) pt->right = node;
            pt = node;
            auto left = node->left;
            auto right = node->right;
            node->left = nullptr;
            node->right = nullptr;
            f(f, left);
            f(f, right);
        };
        f(f, root);
    }

    int minOperations(vector<string> &logs) {
        int depth = 0;
        for (auto &i: logs) {
            if (i == "../") {
                depth = depth - 1 >= 0 ? depth - 1 : 0;
            } else if (i == "./") {
                continue;
            } else {
                ++depth;
            }
        }
        return depth;
    }

    string reverseParentheses(const string &s) {
        vector<char> res;
        stack<int> indices;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '(') {
                indices.push(res.size());
            } else if (s[i] == ')') {
                std::reverse(res.begin() + indices.top(), res.end());
                indices.pop();
            } else {
                res.push_back(s[i]);
            }
        }
        return {res.begin(), res.end()};
    }

    int findTheCity(int n, vector<vector<int>> &edges, int distanceThreshold) {
        vector<pair<int, int>> cities(n);
        vector<vector<int>> to(n, vector<int>(n, std::numeric_limits<int32_t>::max() >> 2));
        for (int i = 0; i < n; ++i) {
            to[i][i] = 0;
        }
        for (auto &i: edges) {
            to[i[0]][i[1]] = i[2];
            to[i[1]][i[0]] = i[2];
        }

        bool flag = true;
        while (flag) {
            flag = false;
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (to[i][j] > to[i][k] + to[k][j]) {
                            flag = true;
                            to[i][j] = to[i][k] + to[k][j];
                        }
                    }
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            cities[i].second = i;
            for (int j = 0; j < n; ++j) {
                if (to[i][j] <= distanceThreshold) {
                    ++cities[i].first;
                    // std::cout << i << " " << j << " " << cities[i].first << std::endl;
                }
            }
        }

        std::sort(cities.begin(), cities.end(), [](auto a, auto b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second > b.second;
        });

        return cities.front().second;
    }

    int minHeightShelves(vector<vector<int>> &books, int shelfWidth) {
        int n = books.size();
        int result = std::numeric_limits<int32_t>::max();

        vector<int> dp(n, -1);

        auto dfs = [&](auto &&dfs, int index) {
            if (index >= n || index < 0) {
                return 0;
            }
            if (dp[index] != -1) {
                return dp[index];
            }

            int curWidth = books[index][0];
            int curHeight = books[index][1];
            dp[index] = dfs(dfs, index + 1) + curHeight;
            int i = index + 1;
            while (i < n) {
                int width = books[i][0];
                int height = books[i][1];
                if (width + curWidth <= shelfWidth) {
                    curHeight = max(curHeight, height);
                    curWidth += width;
                    dp[index] = min(dp[index], curHeight + dfs(dfs, i + 1));
                } else {
                    break;
                }
                ++i;
            }
            return dp[index];
        };
        return dfs(dfs, 0);
    }

    int countSeniors(vector<string> &details) {
        int result = 0;
        for (auto &code: details) {
            if (code[code.size() - 4] > '6' || (code[code.size() - 4] == '6' && code[code.size() - 3] > '0'))
                ++result;
        }
        return result;
    }

    string kthDistinct(vector<string> &arr, int k) {
        unordered_map<string, int> times;
        for (auto &i: arr) {
            ++times[i];
        }
        for (auto &i: arr) {
            if (times[i] == 1) {
                --k;
                if (k <= 0) {
                    return i;
                }
            }
        }
        return "";
    }

    string numberToWords(int num) {

        auto numberToWordsHelper = [&](int num) {
            string digitString[] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
            string teenString[] = {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
            string tenString[] = {"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};

            string result = "";
            if (num > 99) {
                result += digitString[num / 100] + " Hundred ";
            }
            num %= 100;
            if (num < 20 && num > 9) {
                result += teenString[num - 10] + " ";
            } else {
                if (num >= 20) {
                    result += tenString[num / 10] + " ";
                }
                num %= 10;
                if (num > 0) {
                    result += digitString[num] + " ";
                }
            }
            return result;
        };

        if (num == 0) return "Zero";
        string bigString[] = {"Thousand", "Million", "Billion"};
        string result = numberToWordsHelper(num % 1000);
        num /= 1000;
        for (int i = 0; i < 3; ++i) {
            if (num > 0 && num % 1000 > 0) {
                result = numberToWordsHelper(num % 1000) + bigString[i] + " " + result;
            }
            num /= 1000;
        }
        return result.empty() ? result : result.substr(0, result.size() - 1);// Trim trailing space
    }

    vector<vector<int>> spiralMatrixIII(int rows, int cols, int rStart, int cStart) {
        int len = 1;
        bool flag = false;
        int x = rStart;
        int y = cStart;
        int k = 1;
        int direction = 1;
        vector<vector<int>> result;
        result.push_back({x, y});
        int remaining = rows * cols - 1;
        while (remaining) {
            switch (direction) {
                case 0:
                    --x;
                    break;
                case 1:
                    ++y;
                    break;
                case 2:
                    ++x;
                    break;
                case 3:
                    --y;
                    break;
            };
            --k;
            if (k == 0) {
                k = len;
                direction = (direction + 1) % 4;
                if (flag) {
                    flag = false;
                } else {
                    flag = true;
                    ++len;
                }
            }
            if (x >= 0 && x < rows && y >= 0 && y < cols) {
                --remaining;
                result.push_back({x, y});
            }
        }
        return result;
    }

    int numMagicSquaresInside(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.begin()->size();
        int result = 0;
        vector<vector<int>> row_sums(n, vector<int>(m, -1));
        vector<vector<int>> col_sums(n, vector<int>(m, -1));
        vector<vector<int>> diag_sums(n, vector<int>(m, -1));
        vector<vector<int>> diag2_sums(n, vector<int>(m, -1));

        auto sum_row = [&](int x, int y) {
            if (x < 0 || x >= n || y < 0 || y + 2 >= m)
                return -1;
            if (row_sums[x][y] != -1) {
                return row_sums[x][y];
            }
            return row_sums[x][y] = (grid[x][y] + grid[x][y + 1] + grid[x][y + 2]);
        };

        auto sum_col = [&](int x, int y) {
            if (x < 0 || x + 2 >= n || y < 0 || y >= m)
                return -1;
            if (col_sums[x][y] != -1) {
                return col_sums[x][y];
            }
            return col_sums[x][y] = (grid[x][y] + grid[x + 1][y] + grid[x + 2][y]);
        };

        auto distinct = [&](int x, int y) {
            unordered_set<int> hashset;

            constexpr int dx[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
            constexpr int dy[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

            for (int i = 0; i < 9; ++i) {
                int xx = x + dx[i];
                int yy = y + dy[i];
                if (hashset.count(grid[xx][yy])) {
                    return false;
                }
                if (grid[xx][yy] >= 10 || grid[xx][yy] <= 0) {
                    return false;
                }
                hashset.insert(grid[xx][yy]);
            }
            return true;
        };

        auto sum_diag = [&](int x, int y) {
            constexpr int dx[3] = {0, 1, 2};
            constexpr int dy[3] = {2, 1, 0};

            int sum = 0;

            for (int i = 0; i < 3; ++i) {
                int xx = x + dx[i];
                int yy = y + dy[i];
                sum += grid[xx][yy];
            }
            if (diag_sums[x][y] != -1) {
                return diag_sums[x][y];
            }
            return diag_sums[x][y] = sum;
        };

        auto sum_diag2 = [&](int x, int y) {
            constexpr int dx[3] = {0, 1, 2};
            constexpr int dy[3] = {0, 1, 2};

            int sum = 0;

            for (int i = 0; i < 3; ++i) {
                int xx = x + dx[i];
                int yy = y + dy[i];
                sum += grid[xx][yy];
            }

            if (diag2_sums[x][y] != -1) {
                return diag2_sums[x][y];
            }

            return diag2_sums[x][y] = sum;
        };


        for (int i = 0; i < n - 2; ++i) {
            for (int j = 0; j < m - 2; ++j) {
                if (!distinct(i, j)) {
                    continue;
                }
                int sum = sum_diag(i, j);
                if (sum_diag2(i, j) != sum) {
                    continue;
                }
                if (sum_col(i, j) != sum || sum_col(i, j + 1) != sum || sum_col(i, j + 2) != sum) {
                    continue;
                }
                if (sum_row(i, j) != sum || sum_row(i + 1, j) != sum || sum_row(i + 2, j) != sum) {
                    continue;
                }
                ++result;
            }
        }

        return result;
    }

    vector<vector<int>> combinationSum2(vector<int> &candidates, int target) {
        int n = candidates.size();
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> result;
        vector<int> temp;
        int sum = 0;
        auto dfs = [&](auto &&dfs, int index) -> void {
            if (index >= n)
                return;
            for (int i = index; i < n; i++) {
                if (i > index && candidates[i] == candidates[i - 1]) continue;
                if (candidates[i] > target) break;
                temp.push_back(candidates[i]);
                sum += candidates[i];
                if (sum == target) {
                    result.push_back(temp);
                }
                if (sum < target) {
                    dfs(dfs, i + 1);
                }
                temp.pop_back();
                sum -= candidates[i];
            }
        };
        dfs(dfs, 0);
        return result;
    }

    int smallestDistancePair(vector<int> &nums, int k) {
        std::sort(nums.begin(), nums.end());
        int low = 0, high = nums.back() - nums.front();

        auto countPairs = [&](int distance) {
            int count = 0, left = 0;
            for (int right = 1; right < nums.size(); ++right) {
                while (nums[right] - nums[left] > distance) {
                    ++left;
                }
                count += right - left;
            }
            return count;
        };

        while (low < high) {
            int mid = low + (high - low) / 2;
            if (countPairs(mid) < k) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        return low;
    }

    int minSteps(int n) {
        if (n == 1) {
            return 0;
        }
        for (int i = 2; i <= n; ++i) {
            if (n / i * i == n) {
                return i + minSteps(n / i);
            }
        }
        return 0;
    }

    double maxProbability(int n, vector<vector<int>> &edges, vector<double> &succProb, int start_node, int end_node) {
        vector<double> probs(n, 0);
        vector<vector<pair<int, double>>> next(n);
        int m = edges.size();
        for (int i = 0; i < m; ++i) {
            int from = edges[i][0];
            int to = edges[i][1];
            double prob = succProb[i];
            next[from].emplace_back(to, prob);
            next[to].emplace_back(from, prob);
        }
        priority_queue<pair<double, int>> pq;
        pq.emplace(1, start_node);
        while (pq.size()) {
            auto [prob, node] = pq.top();
            pq.pop();
            if (node == end_node) {
                return prob;
            }
            if (probs[node] >= prob) {
                continue;
            }
            probs[node] = prob;
            for (auto i: next[node]) {
                pq.emplace(i.second * prob, i.first);
            }
        }
        return 0;
    }

    int removeStones(vector<vector<int>> &stones) {
        int n = stones.size();
        unordered_map<int, vector<int>> x_hashmap;
        unordered_map<int, vector<int>> y_hashmap;

        for (int i = 0; i < n; ++i) {
            x_hashmap[stones[i][0]].push_back(i);
            y_hashmap[stones[i][1]].push_back(i);
        }
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

        for (auto &k: x_hashmap) {
            int s = k.second.size();
            for (int i = 1; i < s; ++i) {
                uf_union(k.second[0], k.second[i]);
            }
        }

        for (auto &k: y_hashmap) {
            int s = k.second.size();
            for (int i = 1; i < s; ++i) {
                uf_union(k.second[0], k.second[i]);
            }
        }

        unordered_set<int> groups;
        for (int i = 0; i < n; ++i) {
            groups.insert(uf_find(i));
        }

        return n - groups.size();
    }

    int chalkReplacer(vector<int> &chalk, long long int k) {
        int n = chalk.size();
        long long int sum = chalk[0];
        for (int i = 1; i < n; ++i) {
            sum += chalk[i];
        }
        if (k > sum) {
            k = k % sum;
        }
        for (int i = 0; i < n; ++i) {
            k -= chalk[i];
            if (k < 0) {
                return i;
            }
        }
        return 0;
    }

    int getLucky(string s, int k) {
        string number = "";
        for (char x: s) {
            number += to_string(x - 'a' + 1);
        }
        while (k--) {
            int temp = 0;
            for (char x: number) {
                temp += x - '0';
            }
            number = to_string(temp);
        }
        return stoi(number);
    }

    int robotSim(vector<int> &commands, vector<vector<int>> &obstacles) {
        int dx[4] = {0, 1, 0, -1};
        int dy[4] = {1, 0, -1, 0};
        int direction = 0;
        int x = 0;
        int y = 0;
        int res = 0;

        unordered_set<int> obstacle_set;
        for (auto &i: obstacles) {
            obstacle_set.emplace(i[0] * 40000 + i[1]);
        }

        for (auto i: commands) {
            if (i == -1) {
                direction = (direction + 1) % 4;
            } else if (i == -2) {
                direction = (direction + 3) % 4;
            } else {
                int remaining = i;
                while (remaining--) {
                    int xx = x + dx[direction];
                    int yy = y + dy[direction];
                    if (!obstacle_set.count(xx * 40000 + yy)) {
                        x = xx;
                        y = yy;
                    }
                    res = max(res, x * x + y * y);
                }
            }
        }
        return res;
    }

    vector<int> missingRolls(vector<int> &rolls, int mean, int n) {
        int m = rolls.size();
        int sum = mean * (m + n);
        for (int i = 0; i < m; ++i) {
            sum -= rolls[i];
        }
        if (sum < n || sum > 6 * n) {
            return {};
        }
        int initial = sum / n;
        vector<int> res(n, initial);
        sum -= n * initial;
        int k = 0;
        while (sum--) {
            res[k++]++;
            k = k % n;
        }
        return res;
    }

    ListNode *modifiedList(vector<int> &nums, ListNode *head) {
        unordered_set<int> nums_set;
        for (auto i: nums) {
            nums_set.insert(i);
        }
        ListNode *vir = new ListNode(0);
        vir->next = head;
        auto pt = vir;
        while (pt) {
            if (pt->next == nullptr) {
                break;
            }
            while (pt->next && nums_set.count(pt->next->val)) {
                auto new_next = pt->next->next;
                pt->next->next = nullptr;
                pt->next = new_next;
            }
            pt = pt->next;
        }
        return vir->next;
    }

    bool isSubPath(ListNode *head, TreeNode *root) {
        auto dfs = [&](auto &&dfs, TreeNode *node, ListNode *now) -> bool {
            if (node == nullptr && now != nullptr) {
                return false;
            }
            if (now == nullptr) {
                return true;
            }
            if (node->val == now->val) {
                return dfs(dfs, node->left, now->next) || dfs(dfs, node->right, now->next);
            } else {
                return false;
            }
        };
        auto f = [&](auto &&f, TreeNode *node) {
            if (node == nullptr) {
                return false;
            }
            if (dfs(dfs, node, head)) {
                return true;
            } else {
                return f(f, node->left) || f(f, node->right);
            }
        };
        return f(f, root);
    }

    vector<ListNode *> splitListToParts(ListNode *head, int k) {
        auto split = [&](ListNode *node) {
            // split node node->next;
            if (node == nullptr) {
                return;
            }
            node->next = nullptr;
        };

        auto get_size = [&](ListNode *node) {
            int len = 0;
            auto pt = node;
            while (pt) {
                ++len;
                pt = pt->next;
            }
            return len;
        };

        vector<ListNode *> res;

        int len = get_size(head);

        int sublen = len / k;

        int more = len - sublen * k;
        auto pt = head;
        while (more--) {
            res.push_back(pt);
            int split_len = sublen + 1;
            while (--split_len && pt) {
                pt = pt->next;
            }
            auto next = pt->next;
            split(pt);
            pt = next;
        }
        int less = k - len + sublen * k;
        while (--less) {
            res.push_back(pt);
            int split_len = sublen;
            while (--split_len && pt) {
                pt = pt->next;
            }
            if (pt) {
                auto next = pt->next;
                split(pt);
                pt = next;
            }
        }
        if (res.size() < k) {
            res.push_back(pt);
        }
        return res;
    }

    vector<vector<int>> spiralMatrix(int m, int n, ListNode *head) {
        vector<vector<int>> res(m, vector<int>(n, -1));
        int dy[4] = {1, 0, -1, 0};
        int dx[4] = {0, 1, 0, -1};
        int direction = 0;
        int x = 0;
        int y = 0;

        auto move = [&]() {
            int xx = x + dx[direction];
            int yy = y + dy[direction];
            while (xx >= m || xx < 0 || yy >= n || yy < 0 || res[xx][yy] != -1) {
                direction = (direction + 1) % 4;
                xx = x + dx[direction];
                yy = y + dy[direction];
            }
            x = xx;
            y = yy;
        };
        if (head) {
            res[0][0] = head->val;
        } else {
            return res;
        }
        auto pt = head->next;
        while (pt) {
            move();
            res[x][y] = pt->val;
            pt = pt->next;
        }
        return res;
    }

    ListNode *insertGreatestCommonDivisors(ListNode *head) {
        auto gcd = [&](auto &&gcd, int a, int b) -> int {
            if (a > b) {
                return gcd(gcd, a - b, b);
            } else if (a < b) {
                return gcd(gcd, b, a);
            } else
                return a;
        };

        auto insert_gcd = [&](ListNode *node) {
            if (node == nullptr || node->next == nullptr) {
                return;
            }

            int val = gcd(gcd, node->val, node->next->val);
            auto next = node->next;
            node->next = new ListNode(val, next);
        };

        auto pt = head;
        while (pt) {
            insert_gcd(pt);
            if (pt)
                pt = pt->next;
            else
                return head;
            if (pt)
                pt = pt->next;
            else
                return head;
        }
        return head;
    }

    int minBitFlips(int start, int goal) {
        std::bitset<32> start_bit(start);
        std::bitset<32> end_bit(goal);
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            if (start_bit[i] != end_bit[i]) {
                ++res;
            }
        }
        return res;
    }

    int countConsistentStrings(const string &allowed, vector<string> &words) {
        unordered_set<char> allowed_set;
        for (auto i: allowed) {
            allowed_set.insert(i);
        }
        int res = 0;
        for (auto &i: words) {
            bool flag = true;

            for (auto k: i) {
                if (!allowed_set.count(k)) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                res++;
        }
        return res;
    }

    vector<int> xorQueries(vector<int> &arr, vector<vector<int>> &queries) {
        int n = arr.size();
        vector<int> xorPrefix(n, 0);
        xorPrefix[0] = arr[0];
        for (int i = 1; i < n; ++i) {
            xorPrefix[i] = xorPrefix[i - 1] ^ arr[i];
        }
        vector<int> res;
        res.reserve(queries.size());
        for (auto &i: queries) {
            if (i[0] == 0)
                res.push_back(xorPrefix[i[1]]);
            else {
                res.push_back(xorPrefix[i[1]] ^ xorPrefix[i[0] - 1]);
            }
        }
        return res;
    }

    int findMinDifference(vector<string> &timePoints) {
        int n = timePoints.size();
        auto convet2int = [&](const string &s) {
            return stoi(s.substr(0, 2)) * 60 + stoi(s.substr(3, 2));
        };
        priority_queue<int> times;
        for (auto &i: timePoints) {
            times.push(convet2int(i));
        }
        int prev = -1;
        int max_num = std::numeric_limits<int>::min();
        int min_num = std::numeric_limits<int>::max();
        int res = std::numeric_limits<int>::max();
        while (times.size()) {
            max_num = max(max_num, times.top());
            min_num = min(min_num, times.top());
            if (prev == -1) {
                prev = times.top();
            } else {
                res = min(res, std::abs(prev - times.top()));
                prev = times.top();
            }
            times.pop();
        }
        res = min(res, min_num + 24 * 60 - max_num);
        return res;
    }

    vector<string> uncommonFromSentences(const string &s1, const string &s2) {
        unordered_map<string, int> s1_times;
        vector<string> res;
        int index = 0;
        while (index < s1.size()) {
            auto next_ = std::find(s1.begin() + index + 1, s1.end(), ' ');
            int next_index = next_ - s1.begin();
            s1_times[s1.substr(index, next_index - index)]++;
            index = next_index + 1;
        }
        unordered_map<string, int> s2_times;
        index = 0;
        while (index < s2.size()) {
            auto next_ = std::find(s2.begin() + index + 1, s2.end(), ' ');
            int next_index = next_ - s2.begin();
            if (s1_times.count(s2.substr(index, next_index - index))) {
                s1_times[s2.substr(index, next_index - index)] = -1;
            } else {
                s2_times[s2.substr(index, next_index - index)]++;
            }
            index = next_index + 1;
        }
        for (auto &i: s1_times) {
            if (i.second == 1) {
                res.emplace_back(i.first);
            }
        }
        for (auto &i: s2_times) {
            if (i.second == 1) {
                res.emplace_back(i.first);
            }
        }
        return res;
    }

    vector<int> diffWaysToCompute(const string &expression) {
        vector<int> res;
        for (int i = 0; i < expression.size(); ++i) {
            char oper = expression[i];
            if (oper == '+' || oper == '-' || oper == '*') {
                vector<int> s1 = diffWaysToCompute(expression.substr(0, i));
                vector<int> s2 = diffWaysToCompute(expression.substr(i + 1));
                for (int a: s1) {
                    for (int b: s2) {
                        if (oper == '+') res.push_back(a + b);
                        else if (oper == '-')
                            res.push_back(a - b);
                        else if (oper == '*')
                            res.push_back(a * b);
                    }
                }
            }
        }
        if (res.empty()) res.push_back(stoi(expression));
        return res;
    }

    string shortestPalindrome(const string &s) {
        if (s.empty()) return "";
        string reverse_s(s.rbegin(), s.rend());
        string combined = s + "#" + reverse_s;
        vector<int> prefix(combined.size(), 0);
        for (int i = 1; i < combined.size(); ++i) {
            int j = prefix[i - 1];
            while (j > 0 && combined[i] != combined[j]) {
                j = prefix[j - 1];
            }
            if (combined[i] == combined[j]) {
                ++j;
            }
            prefix[i] = j;
        }
        int longest_palindrome_len = prefix.back();
        return reverse_s.substr(0, s.size() - longest_palindrome_len) + s;
    }

    vector<int> lexicalOrder(int n) {
        string ns = to_string(n);
        string temp = "";
        vector<int> res;
        auto dfs = [&](auto &&dfs, int depth) -> void {
            if (temp.size()) {
                int num = stoi(temp);
                if (num <= n) {
                    res.push_back(num);
                    for (char c = '0'; c <= '0' + 9; ++c) {
                        temp.push_back(c);
                        dfs(dfs, depth + 1);
                        temp.pop_back();
                    }
                }
            } else {
                for (char c = '1'; c <= '0' + 9; ++c) {
                    temp.push_back(c);
                    dfs(dfs, depth + 1);
                    temp.pop_back();
                }
            }
        };
        dfs(dfs, 0);
        return res;
    }

    int minExtraChar(const string &s, vector<string> &dictionary) {
        Trie trie;
        for (auto &i: dictionary) {
            trie.add(i);
        }
        int n = s.size();
        vector<int> dp(n, -1);
        auto dfs = [&](auto &&dfs, int index) -> int {
            if (index >= n || index < 0) {
                return 0;
            }
            if (dp[index] != -1) {
                return dp[index];
            }
            dp[index] = 1 + dfs(dfs, index + 1);
            for (int len = 1; index + len <= n; ++len) {
                if (trie.contains(s.substr(index, len))) {
                    dp[index] = min(dp[index], dfs(dfs, index + len));
                }
            }
            return dp[index];
        };
        return dfs(dfs, 0);
    }

    int longestCommonPrefix(vector<int> &arr1, vector<int> &arr2) {
        PrefixTrie prefix_trie;
        int res = 0;
        for (auto i: arr2) {
            prefix_trie.add(to_string(i));
        }
        for (auto i: arr1) {
            string temp = to_string(i);
            if (temp.size() <= res) {
                continue;
            }
            string prefix = temp.substr(0, 1);
            if (prefix_trie.contains(prefix)) {
                res = max(res, 1);
            }
            for (int len = 1; len < temp.size(); ++len) {
                prefix.push_back(temp[len]);
                if (prefix_trie.contains(prefix)) {
                    res = max(res, len + 1);
                } else {
                    break;
                }
            }
        }
        return res;
    }

    template<char min_char, char max_char>
    struct PrefixNumTrieNode {
        PrefixNumTrieNode *next[max_char - min_char + 1] = {nullptr};
        int isEnd = 0;
        PrefixNumTrieNode() {
            for (int i = 0; i <= max_char - min_char; ++i) {
                next[i] = nullptr;
            }
        }
    };

    template<char min_char, char max_char>
    struct PrefixNumTrie {
        using Node = PrefixNumTrieNode<min_char, max_char>;
        Node *root = new Node();
        int contains(const string &input) {
            auto temp = root;
            for (char ch: input) {
                int idx = ch - min_char;
                temp = temp->next[idx];
                if (temp == nullptr) {
                    return false;
                }
            }
            return temp->isEnd;
        }

        void add(const string &input) {
            auto temp = root;
            for (char ch: input) {
                int idx = ch - min_char;
                if (temp->next[idx] == nullptr)
                    temp->next[idx] = new Node();
                temp->isEnd += 1;
                temp = temp->next[idx];
            }
            temp->isEnd += 1;
        }

        int sum(const string &input) {
            int res = 0;
            auto temp = root;
            for (size_t i = 0; i < input.size(); ++i) {
                int idx = input[i] - min_char;
                if (temp->next[idx] == nullptr)
                    return res;
                if (i != 0)
                    res += temp->isEnd;
                temp = temp->next[idx];
            }
            res += temp->isEnd;
            return res;
        }
    };

    vector<int> sumPrefixScores(vector<string> &words) {
        int n = words.size();
        vector<int> res;
        res.reserve(n);
        PrefixNumTrie<'a', 'z'> prefix_trie;
        for (auto &i: words) {
            prefix_trie.add(i);
        }
        for (auto &i: words) {
            res.push_back(prefix_trie.sum(i));
        }
        return res;
    }

    bool canArrange(vector<int> &arr, int k) {
        vector<int> nums(k, 0);
        for (auto i: arr) {
            nums[(k + (i % k)) % k]++;
        }
        if (nums[0] & 1) {
            return false;
        }
        for (int i = 1; i < k; ++i) {
            if (nums[i] != nums[k - i]) {
                return false;
            }
        }
        return true;
    }

    vector<int> arrayRankTransform(vector<int> &arr) {
        unordered_map<int, int> to_rank;
        set<int> number_set;
        for (auto i: arr) {
            number_set.insert(i);
        }
        int rank = 1;
        for (auto i: number_set) {
            to_rank[i] = rank++;
        }
        for (auto &i: arr) {
            i = to_rank[i];
        }
        return arr;
    }

    long long dividePlayers(vector<int> &skill) {
        int n = skill.size();
        int totalSkill = 0;
        vector<int> skillFrequency(1001, 0);
        for (int playerSkill: skill) {
            totalSkill += playerSkill;
            skillFrequency[playerSkill]++;
        }
        if (totalSkill % (n / 2) != 0) {
            return -1;
        }
        int targetTeamSkill = totalSkill / (n / 2);
        long long totalChemistry = 0;
        for (int playerSkill: skill) {
            int partnerSkill = targetTeamSkill - playerSkill;
            if (skillFrequency[partnerSkill] == 0) {
                return -1;
            }
            totalChemistry += (long long) playerSkill * (long long) partnerSkill;
            skillFrequency[partnerSkill]--;
        }
        return totalChemistry / 2;
    }

    int minLength(const string &s) {
        stack<char> stack;
        for (int i = 0; i < s.length(); i++) {
            char cur_char = s[i];
            if (stack.empty()) {
                stack.push(cur_char);
                continue;
            }
            if (cur_char == 'B' && stack.top() == 'A') {
                stack.pop();
            } else if (cur_char == 'D' && stack.top() == 'C') {
                stack.pop();
            } else {
                stack.push(cur_char);
            }
        }
        return stack.size();
    }

    int minSwaps(const string &s) {
        int num = 0;
        for (auto &i: s) {
            if (i == '[') {
                ++num;
            } else if (num > 0) {
                --num;
            }
        }
        return (num + 1) / 2;
    }

    int minAddToMakeValid(const string &s) {
        int num = 0;
        int res = 0;
        for (auto &i: s) {
            if (i == '(') {
                ++num;
            } else if (num > 0) {
                --num;
            } else {
                ++res;
            }
        }
        return res + num;
    }

    int maxWidthRamp(vector<int> &nums) {
        int res = 0;
        vector<int> candidates;
        for (int i = 0; i < nums.size(); ++i) {
            if (candidates.empty()) {
                candidates.push_back(i);
            } else {
                if (nums[candidates.back()] > nums[i]) {
                    candidates.push_back(i);
                }
            }
        }
        for (int i = nums.size() - 1; i >= 0; --i) {
            while (candidates.size() && nums[candidates.back()] <= nums[i]) {
                res = max(res, i - candidates.back());
                candidates.pop_back();
            }
        }
        return res;
    }

    int minGroups(vector<vector<int>> &intervals) {
        vector<pair<int, int>> events;
        for (auto &interval: intervals) {
            events.emplace_back(interval[0], 1);
            events.emplace_back(interval[1] + 1, -1);
        }
        std::sort(events.begin(), events.end());
        int activeGroups = 0;
        int maxGroups = 0;
        for (auto &event: events) {
            activeGroups += event.second;
            maxGroups = std::max(maxGroups, activeGroups);
        }
        return maxGroups;
    }

    long long maxKelements(vector<int> &nums, int k) {
        long long int res = 0;
        priority_queue<int> pq;
        for (auto i: nums) {
            pq.push(i);
        }
        while (k--) {
            int score = pq.top();
            pq.pop();
            res += score;
            pq.push((score + 2) / 3);
        }
        return res;
    }

    long long minimumSteps(const string &s) {
        int n = s.size();
        int i = 0;
        int j = n - 1;
        long long res = 0;
        while (i < j) {
            while (i < j && s[i] == '0') {
                ++i;
            }
            while (i < j && s[j] == '1') {
                --j;
            }
            if (i != j) {
                res += j - i;
                ++i;
                --j;
            } else {
                break;
            }
        }
        return res;
    }

    string longestDiverseString(int a, int b, int c) {
        int currA = 0, currB = 0, currC = 0;
        int maxLen = a + b + c, i = 0;
        std::string result;
        while (i < maxLen) {
            if ((currA != 2 && a >= b && a >= c) || (a > 0 && (currB == 2 || currC == 2))) {
                result += 'a';
                currA++;
                currB = 0;
                currC = 0;
                a--;
            } else if ((currB != 2 && b >= a && b >= c) || (b > 0 && (currA == 2 || currC == 2))) {
                result += 'b';
                currB++;
                currA = 0;
                currC = 0;
                b--;
            } else if ((currC != 2 && c >= a && c >= b) || (c > 0 && (currA == 2 || currB == 2))) {
                result += 'c';
                currC++;
                currA = 0;
                currB = 0;
                c--;
            }
            i++;
        }
        return result;
    }

    int countMaxOrSubsets(vector<int> &nums) {
        int max_or = 0;
        for (auto i: nums) {
            max_or |= i;
        }
        int res = 0;
        auto dfs = [&](auto &&dfs, int or_sum, int index) {
            if (index >= nums.size()) {
                if (or_sum == max_or) {
                    ++res;
                }
                return;
            }
            dfs(dfs, or_sum, index + 1);
            or_sum |= nums[index];
            if (or_sum == max_or) {
                res += (1 << (nums.size() - index - 1));
            } else
                dfs(dfs, or_sum, index + 1);
        };
        dfs(dfs, 0, 0);
        return res;
    }

    int maximumSwap(int num) {
        int base[10] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
        if (num == 100000000) {
            return num;
        }
        int res = 0;
        for (int i = 0; num / base[i] >= 1; ++i) {
            for (int j = 0; num / base[j] >= 1; ++j) {
                if (i == j)
                    continue;
                int basei = base[i];
                int basej = base[j];
                int numi = (num / basei) % 10;
                int numj = (num / basej) % 10;
                int delta = numi * basej + numj * basei - numi * basei - numj * basej;
                res = std::max(res, delta);
            }
        }
        return res + num;
    }

    char findKthBit(int n, int k) {
        if (n == 1 && k == 1) {
            return '0';
        }
        int len = (1 << n) - 1;
        int mid = (len + 1) >> 1;
        if (k == mid)
            return '1';
        if (k > mid) {
            return '0' + '1' - findKthBit(n - 1, (1 << n) - k);
        } else {
            return findKthBit(n - 1, k);
        }
    }

    int maxUniqueSplit(const string &s) {
        set<string> strings;
        string temp;
        int res = 1;
        auto dfs = [&](auto &&dfs, int index) {
            if (index >= s.size()) {
                if (temp.size()) {
                    if (strings.count(temp)) {
                        return;
                    } else {
                        strings.insert(temp);
                    }
                }
                res = max(res, static_cast<int>(strings.size()));
                if (temp.size()) {
                    strings.erase(temp);
                }
                return;
            }
            temp.push_back(s[index]);
            dfs(dfs, index + 1);
            if (!strings.count(temp)) {
                strings.insert(temp);
                string backup;
                backup.swap(temp);
                dfs(dfs, index + 1);
                strings.erase(backup);
                backup.swap(temp);
            }
            temp.pop_back();
        };
        dfs(dfs, 0);
        return res;
    }

    bool parseBoolExpr(string expression) {
        stack<char> operators;
        stack<char> operands;
        for (auto i: expression) {
            if (i == ')') {
                char op = operators.top();
                operators.pop();
                bool res;
                switch (op) {
                    case '|':
                        res = false;
                        while (operands.size() && operands.top() != '(') {
                            res |= (operands.top() == 't');
                            operands.pop();
                        }
                        if (operands.top() == '(')
                            operands.pop();
                        operands.push((res ? 't' : 'f'));
                        break;
                    case '&':
                        res = true;
                        while (operands.size() && operands.top() != '(') {
                            res &= (operands.top() == 't');
                            operands.pop();
                        }
                        if (operands.top() == '(')
                            operands.pop();
                        operands.push((res ? 't' : 'f'));
                        break;
                    case '!':
                        res = (operands.top() == 't' ? false : true);
                        operands.pop();
                        if (operands.top() == '(')
                            operands.pop();
                        operands.push((res ? 't' : 'f'));
                        break;
                }
            } else if (i == '(' || i == 't' || i == 'f') {
                operands.push(i);
            } else if (i == '!' || i == '|' || i == '&') {
                operators.push(i);
            }
        }
        return operands.top() == 't' ? true : false;
    }

    long long kthLargestLevelSum(TreeNode *root, int k) {
        unordered_map<int, long long int> level_sums;
        auto dfs = [&](auto &&dfs, TreeNode *node, int level) -> void {
            if (node == nullptr) {
                return;
            }
            level_sums[level] += node->val;
            dfs(dfs, node->left, level + 1);
            dfs(dfs, node->right, level + 1);
        };
        dfs(dfs, root, 0);
        if (level_sums.size() < k) {
            return -1;
        }
        priority_queue<long long int> pq;
        for (auto i: level_sums) {
            pq.push(-i.second);
            while (pq.size() > k) {
                pq.pop();
            }
        }
        return -pq.top();
    }

    TreeNode *replaceValueInTree(TreeNode *root) {
        if (!root) return nullptr;
        queue<TreeNode *> q;
        int prev = root->val;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            int curr = 0;
            while (size > 0) {
                TreeNode *temp = q.front();
                q.pop();
                int leftRight = (temp->left ? temp->left->val : 0) + (temp->right ? temp->right->val : 0);
                if (temp->left) {
                    temp->left->val = leftRight;
                    q.push(temp->left);
                }
                if (temp->right) {
                    temp->right->val = leftRight;
                    q.push(temp->right);
                }
                curr += leftRight;
                temp->val = prev - temp->val;
                size--;
            }
            prev = curr;
        }
        return root;
    }

    bool flipEquiv(TreeNode *root1, TreeNode *root2) {
        if (root1 == nullptr && root2 == nullptr) {
            return true;
        }
        if (root1 == nullptr || root2 == nullptr) {
            return false;
        }
        if (root1->val == root2->val) {
            return flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left) || flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right);
        } else {
            return false;
        }
    }

    int countSquares(vector<vector<int>> &matrix) {
        int n = matrix.size();
        int m = matrix.begin()->size();
        vector<vector<int>> dp(n, vector<int>(m, -1));
        auto f = [&](auto &&f, int i, int j) {
            if (i < 0 || j < 0 || matrix[i][j] == 0) return 0;
            if (dp[i][j] != -1) return dp[i][j];
            return dp[i][j] = 1 + min({f(f, i - 1, j), f(f, i - 1, j - 1), f(f, i, j - 1)});
        };
        int res = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                res += f(f, i, j);
        return res;
    }

    int maxMoves(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.begin()->size();
        vector<vector<int>> dp(n, vector<int>(m, -1));
        auto f = [&](auto &&f, int x, int y) {
            if (y + 1 >= m) {
                return 0;
            }
            if (dp[x][y] != -1) {
                return dp[x][y];
            }
            int res = 0;
            if (x - 1 >= 0) {
                res = max(res, (grid[x][y] < grid[x - 1][y + 1]) ? f(f, x - 1, y + 1) + 1 : 0);
            }
            res = max(res, (grid[x][y] < grid[x][y + 1]) ? f(f, x, y + 1) + 1 : 0);

            if (x + 1 < n) {
                res = max(res, (grid[x][y] < grid[x + 1][y + 1]) ? f(f, x + 1, y + 1) + 1 : 0);
            }
            return dp[x][y] = res;
        };
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res = max(res, f(f, i, 0));
        }
        return res;
    }

    bool isCircularSentence(const string &sentence) {
        char c = *sentence.begin();
        int index = 0;
        while (index < sentence.size()) {
            if (sentence[index] == ' ') {
                if (index + 1 < sentence.size()) {
                    if (sentence[index - 1] != sentence[index + 1]) {
                        return false;
                    }
                }
            }
            if (index == sentence.size() - 1) {
                if (c != sentence[index]) {
                    return false;
                }
            }
            ++index;
        }
        return true;
    }

    bool rotateString(const string &s, const string &goal) {
        if (s.size() != goal.size()) {
            return false;
        }
        string tmp = s + s;
        return tmp.find(goal) < s.size();
    }

    string compressedString(const string &word) {
        string res = "";
        if (word.empty()) return res;
        char c = word[0];
        int times = 0;
        for (int i = 0; i < word.size(); ++i) {
            if (times < 9 && word[i] == c) {
                ++times;
            } else {
                res.push_back('0' + times);
                res.push_back(c);
                times = 1;
                c = word[i];
            }
        }
        res.push_back('0' + times);
        res.push_back(c);
        return res;
    }

    int minChanges(const string &s) {
        char c = s[0];
        int times = 1;
        int res = 0;
        for (int i = 1; i < s.size(); ++i) {
            if (s[i] == c) {
                ++times;
            } else {
                if (times & 1) {
                    ++res;
                    ++times;
                } else {
                    times = 1;
                    c = s[i];
                }
            }
        }
        return res;
    }

    bool canSortArray(vector<int> &nums) {
        auto numberOfSetBits = [&](int num) {
            bitset<32> bits(num);
            return bits.count();
        };
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[j] < nums[i]) {
                    if (numberOfSetBits(nums[i]) != numberOfSetBits(nums[j])) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    int largestCombination(vector<int> &candidates) {
        vector<int> hm(32, 0);
        int res = 0;
        for (auto i: candidates) {
            bitset<32> bits(i);
            for (int k = 0; k < 32; ++k) {
                if (bits.test(k)) {
                    hm[k]++;
                }
            }
        }
        return *std::max_element(hm.begin(), hm.end());
    }

    vector<int> getMaximumXor(vector<int> &nums, int maximumBit) {
        int n = nums.size();
        vector<int> prefix_xor_sum(n, 0);
        prefix_xor_sum[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            prefix_xor_sum[i] = prefix_xor_sum[i - 1] ^ nums[i];
        }
        vector<int> res(n, (1 << maximumBit) - 1);
        for (int i = 0; i < n; ++i) {
            res[n - 1 - i] = (res[n - 1 - i] ^ prefix_xor_sum[i]) & ((1 << maximumBit) - 1);
        }
        return res;
    }

    long long minEnd(int n, int x) {
        std::bitset<64> bits(x);
        if (n == 1) {
            return x;
        }
        std::bitset<64> n_bits(n - 1);
        int index = 0;
        for (int i = 0; i < 64; ++i) {
            if (!bits.test(i)) {
                if (n_bits.test(index++))
                    bits.set(i);
            }
        }
        return bits.to_ullong();
    }

    int minimumSubarrayLength(vector<int> &nums, int k) {
        int res = std::numeric_limits<int>::max();
        bitset<32> k_bits(k);
        vector<int> bits(32, 0);
        int left = 0;
        int right = 0;
        if (k == 0) return 1;
        auto f = [&]() {
            int sum = 0;
            for (int i = 0; i < 32; ++i) {
                sum |= bits[i] > 0 ? (1 << i) : 0;
            }
            return sum >= k;
        };
        while (right < nums.size()) {
            if (left == right || !f()) {
                for (int i = 0; i < 32; ++i) {
                    if (nums[right] & (1 << i)) {
                        bits[i]++;
                    }
                }
                ++right;
                if (f()) {
                    res = min(res, right - left);
                }
            } else {
                for (int i = 0; i < 32; ++i) {
                    if (nums[left] & (1 << i)) {
                        bits[i]--;
                    }
                }
                ++left;
                if (f()) {
                    res = min(res, right - left);
                }
            }
        }
        while (left < right && f()) {
            res = min(res, right - left);
            for (int i = 0; i < 32; ++i) {
                if (nums[left] & (1 << i)) {
                    bits[i]--;
                }
            }
            ++left;
        }
        return res == std::numeric_limits<int>::max() ? -1 : res;
    }

    bool primeSubOperation(vector<int> &nums) {
        constexpr auto primes = [&]() {
            std::array<int, 168> primes = {};
            primes[0] = 2;
            size_t count = 1;
            for (int x = 3; x < 1000; x += 2) {
                bool is_prime = true;
                for (size_t i = 0; i < count; ++i) {
                    if (primes[i] * primes[i] > x) break;
                    if (x % primes[i] == 0) {
                        is_prime = false;
                        break;
                    }
                }
                if (is_prime) {
                    primes[count++] = x;
                }
            }
            return primes;
        }();

        auto get_prime = [&](int index) {
            if (index >= 0) {
                return primes[index];
            } else {
                return 0;
            }
        };

        auto find_prime = [&](int num) {
            return std::lower_bound(primes.begin(), primes.end(), num) - primes.begin();
        };

        nums[0] -= get_prime(find_prime(nums[0]) - 1);
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] <= nums[i - 1]) {
                return false;
            }
            auto prime_index = find_prime(nums[i] - nums[i - 1]) - 1;
            nums[i] -= get_prime(prime_index);
        }
        return true;
    }

    vector<int> maximumBeauty(vector<vector<int>> &items, vector<int> &queries) {
        std::sort(items.begin(), items.end(), [](const vector<int> &a, const vector<int> &b) {
            return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
        });
        int n = queries.size();
        vector<int> max_beauty(items.size());
        max_beauty[0] = items[0][1];
        for (int i = 1; i < items.size(); ++i) {
            max_beauty[i] = max(max_beauty[i - 1], items[i][1]);
        }
        vector<int> res(n, 0);
        for (auto i = 0; i < n; ++i) {
            auto iter = std::lower_bound(items.begin(), items.end(), vector<int>{queries[i] + 1, 0});
            if (iter == items.begin()) {
                continue;
            } else {
                res[i] = max_beauty[iter - items.begin() - 1];
            }
        }
        return res;
    }

    long long countFairPairs(vector<int> &nums, int lower, int upper) {
        int n = nums.size();
        std::sort(nums.begin(), nums.end());
        long long res = 0;
        for (int i = 0; i < n; ++i) {
            auto &num = nums[i];
            auto iter = nums.begin() + i;
            auto from = std::lower_bound(iter + 1, nums.end(), lower - num) - nums.begin();
            auto to = std::upper_bound(iter + 1, nums.end(), upper - num) - nums.begin();
            res += to - from;
        }
        return res;
    }

    int minimizedMaximum(int n, vector<int> &quantities) {
        long long int res = std::numeric_limits<long long int>::max();

        auto f = [&](long long int candidate) {
            int _n = n;
            for (int i = 0; i < quantities.size(); ++i) {
                int quantity = quantities[i];
                int num = (quantity + candidate - 1) / candidate;
                _n -= num;
                if (_n < 0) {
                    return false;
                }
            }
            res = min(res, candidate);
            return true;
        };

        long long int left = 1;
        long long int right = std::accumulate(quantities.begin(), quantities.end(), 0ll);
        while (left <= right) {
            long long int mid = (left + right) >> 1;
            if (f(mid)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    int findLengthOfShortestSubarray(vector<int> &arr) {
        int n = arr.size();
        int left = 0;
        for (int i = 1; i < n; ++i) {
            if (arr[i] < arr[i - 1]) {
                left = i;
                break;
            }
        }
        if (left == 0) return 0;
        int right = n;
        for (int i = n - 1; i > 0; --i) {
            if (arr[i] < arr[i - 1]) {
                right = i;
                break;
            }
        }
        int res = min(min(n - left, right), n - 1);
        right = n - 1;
        while (left--) {
            if (arr[right] >= arr[left]) {
                res = min(res, right - left - 1);
            } else
                continue;
            while (right > left && arr[right] >= arr[right - 1]) {
                --right;
                if (arr[right] >= arr[left]) {
                    res = min(res, right - left - 1);
                } else
                    break;
            }
        }
        return res;
    }

    vector<int> resultsArray(vector<int> &nums, int k) {
        int n = nums.size();
        vector<int> res(n - k + 1);
        deque<pair<int, int>> dq;
        for (int i = 0; i < k; ++i) {
            int num = nums[i];
            while (dq.size() && num != dq.back().first + 1) {
                dq.pop_back();
            }
            dq.emplace_back(num, i);
        }
        res[0] = (dq.size() == k ? dq.back().first : -1);
        for (int i = k; i < n; ++i) {
            int num = nums[i];
            while (dq.size() && dq.front().second <= i - k) {
                dq.pop_front();
            }
            while (dq.size() && num != dq.back().first + 1) {
                dq.pop_back();
            }
            dq.emplace_back(num, i);
            res[i - k + 1] = (dq.size() == k ? dq.back().first : -1);
        }
        return res;
    }
};

int main() {
    [&out = std::cout]() -> void {
        out << "Hello World\n"
            << std::endl;
    }();
    return 0;
}