//
// Created by Fitzgerald on 2/19/23.
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

class Node {
public:
    bool val;
    bool isLeaf;
    Node *topLeft;
    Node *topRight;
    Node *bottomLeft;
    Node *bottomRight;

    Node() {
        val = false;
        isLeaf = false;
        topLeft = nullptr;
        topRight = nullptr;
        bottomLeft = nullptr;
        bottomRight = nullptr;
    }

    Node(bool _val, bool _isLeaf) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = nullptr;
        topRight = nullptr;
        bottomLeft = nullptr;
        bottomRight = nullptr;
    }

    Node(bool _val, bool _isLeaf, Node *_topLeft, Node *_topRight, Node *_bottomLeft, Node *_bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};


struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(nullptr) {}
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
        // Hierholzer's algorithm
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

    ListNode *detectCycle(ListNode *head) {
        //Floyd's Tortoise and Hare
        ListNode *fast = head;
        ListNode *slow = head;
        while (slow != nullptr && fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
            if (fast == slow) {
                while (head != fast) {
                    fast = fast->next;
                    head = head->next;
                }
                return fast;
            }
        }
        return nullptr;
    }

    void sortedListToBSTDFS(TreeNode *root, vector<int> &nums, int begin, int end) {
        if (begin <= end) {
            int mid = (begin + end) / 2;
            root->val = nums[mid];
            if (begin <= mid - 1) {
                if (root->left == nullptr) {
                    root->left = new TreeNode();
                }
                sortedListToBSTDFS(root->left, nums, begin, mid - 1);
            }
            if (mid + 1 <= end) {
                if (root->right == nullptr) {
                    root->right = new TreeNode();
                }
                sortedListToBSTDFS(root->right, nums, mid + 1, end);
            }
        }
    }

    TreeNode *sortedListToBST(ListNode *head) {
        vector<int> nums;
        while (head != nullptr) {
            nums.push_back(head->val);
            head = head->next;
        }
        int n = nums.size();
        if (!n) {
            return nullptr;
        }
        TreeNode *res = new TreeNode();
        sortedListToBSTDFS(res, nums, 0, n - 1);
        return res;
    }

    ListNode *mergeKLists(vector<ListNode *> &lists) {
        auto cmp = [](ListNode *a, ListNode *b) -> bool {
            return a->val > b->val;
        };
        priority_queue<ListNode *, vector<ListNode *>, decltype(cmp)> heap(cmp);
        ListNode *res = nullptr;
        ListNode *cur = res;
        for (auto &i: lists) {
            if (i != nullptr) {
                heap.push(i);
            }
        }
        while (!heap.empty()) {
            auto top = heap.top();
            heap.pop();
            if (top->next) {
                heap.push(top->next);
            }
            if (res == nullptr) {
                res = new ListNode(top->val);
                cur = res;
            } else {
                cur->next = new ListNode(top->val);
                cur = cur->next;
            }
        }
        return res;
    }

    void subsetsHelper(int i, vector<int> &nums, vector<vector<int>> &ans, vector<int> &temp) {
        if (i == nums.size()) {
            return;
        }
        if (i == 0) {
            ans.push_back(temp);
        }
        for (int j = i; j < nums.size(); j++) {
            temp.push_back(nums[j]);
            ans.push_back(temp);
            subsetsHelper(j + 1, nums, ans, temp);
            temp.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int> &nums) {
        vector<vector<int>> ans;
        vector<int> temp;
        subsetsHelper(0, nums, ans, temp);
        return ans;
    }

//    The best solution
//    This code is a C++ function designed to solve a graph theory problem: given a tree with n nodes (represented by edges), find the diameter of all subtrees and count the occurrences of each diameter. The code employs techniques such as lambda expressions, recursion, bitwise operations, and breadth-first search (BFS).
//    Here is a detailed explanation of the code:
//
//            Define a function called countSubgraphsForEachDiameter, which takes two arguments: an integer n (number of nodes) and a 2D integer vector edges (the tree's edges).
//    Define a vector of type bitset<16> called adj, which represents the tree's adjacency matrix. The tree's node labels range from 1 to n, so the size of adj is n + 1.
//    Iterate through edges, updating adj to populate the adjacency matrix.
//    Define a 16x16 integer array called dis, which stores the distances between any two nodes in the tree.
//    Define a lambda expression called dfs, which calculates the distances between any two nodes in the tree using depth-first search (DFS). dfs takes four arguments: root, u, p, and d. root is the root node of the subtree rooted at u, p is the parent node of u, and d is the distance between root and u.
//    Use dfs to calculate the distances between any two nodes in the tree and store the results in the dis array.
//    Define an integer vector called ans, which stores the number of occurrences of each diameter. Initially, set all elements to 0.
//    Define a type alias called T, of type pair<B, int>. Objects of type T store a subtree (represented as a bitset) and its diameter.
//    Define a vector of type T called V, which stores all visited subtrees and their diameters.
//    Define an integer queue called Q for breadth-first search (BFS). Initially, enqueue node 1.
//    Define a boolean vector called vis, which records whether each node has been visited. Initially, set all elements to false, and mark node 1 as visited.
//    Traverse all nodes of the tree using BFS. For the currently visited node i, perform the following actions:
//            a. Calculate the diameter of all visited subtrees, and update the ans vector.
//    b. Add node i to the current subtree, and append the new subtree and its diameter to the V vector.
//    c. Enqueue all unvisited neighbor nodes of node i into the queue Q, and mark them as visited.
//    Once BFS is complete, return the ans vector, which represents the number of occurrences of each diameter.
//    The primary goal of the code is to find the diameter of all subtrees in a given tree and count their occurrences. The code employs depth-first search (DFS) and breadth-first search (BFS) techniques, as well as bitwise operations to optimize storage and computation.
//
//
//    vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>>& edges) {
//        using B = bitset<16>;
//        vector<B> adj(n + 1);
//        for (auto& edge: edges) {
//            int u = edge[0], v = edge[1];
//            adj[u].set(v);
//            adj[v].set(u);
//        }
//        int dis[16][16];
//        auto dfs = [&](auto&& dfs, int root, int u, int p, int d) -> void {
//            dis[root][u] = d;
//            for (int v = 1; v <= n; v++) {
//                if (!adj[u].test(v)) continue;
//                if (v == p) continue;
//                dfs(dfs, root, v, u, d + 1);
//            }
//        };
//        for (int v = 1; v <= n; v++) dfs(dfs, v, v, v, 0);
//
//        vector<int> ans(n - 1);
//
//        using T = pair<B, int>;
//        vector<T> V;
//        queue<int> Q;
//        vector<bool> vis(16);
//        Q.push(1);
//        vis[1] = true;
//
//        while (!Q.empty()) {
//            auto i = Q.front();
//            Q.pop();
//            auto conn = adj[i];
//            int sz = V.size();
//            for (int j = 0; j < sz; j++) {
//                auto [bs, d] = V[j];
//                if (!(conn & bs).any()) continue;
//                for (int k = 1; k <= n; k++) {
//                    if (!bs.test(k)) continue;
//                    d = max(d, dis[k][i]);
//                }
//                ans[d - 1] += 1;
//                bs.set(i);
//                V.emplace_back(bs, d);
//            }
//            V.emplace_back(B(1 << i), 0);
//            for (int v = 1; v <= n; v++) {
//                if (!adj[i].test(v)) continue;
//                if (vis[v]) continue;
//                vis[v] = true;
//                Q.push(v);
//            }
//        }
//        return ans;
//    }

    int countSubgraphsForEachDiameterDFS(int &maxDis, unordered_map<int, unordered_set<int>> &graph, set<int> &vertices,
                                         int node, int prev) {
        unordered_set<int> &next = graph[node];
        int top1 = 0;
        int top2 = 0;
        for (auto i: next) {
            if (i == prev || vertices.count(i) == 0) {
                continue;
            }
            int len = countSubgraphsForEachDiameterDFS(maxDis, graph, vertices, i, node);
            if (len > top1) {
                top2 = top1;
                top1 = len;
            } else if (len > top2) {
                top2 = len;
            }
        }
        maxDis = max(maxDis, top2 + top1);
        return top1 + 1;
    }

    vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>> &edges) {
        vector<int> res(n - 1, 0);
        unordered_map<int, unordered_set<int>> graph;
        for (auto &i: edges) {
            graph[i[0]].insert(i[1]);
            graph[i[1]].insert(i[0]);
        }
        for (int i = 0; i < (1 << n); ++i) {
            set<int> vertices;
            for (int j = 0; j < n; ++j) {
                if (i & (1 << j)) {
                    vertices.insert(j + 1);
                }
            }
            int numOfEdges = 0;
            for (auto i = vertices.begin(); i != vertices.end(); ++i) {
                for (auto j = i; j != vertices.end(); ++j) {
                    if (*i == *j)
                        continue;
                    if (graph[*i].count(*j))
                        ++numOfEdges;
                }
            }
            if (numOfEdges == 0 || numOfEdges != vertices.size() - 1) {
                continue;
            } else {
                int maxDis = 0;
                countSubgraphsForEachDiameterDFS(maxDis, graph, vertices, *vertices.begin(), 0);
                if (maxDis <= vertices.size())
                    ++res[maxDis - 1];
            }
        }
        return res;
    }

    bool isSymmetric(TreeNode *root) {
        if (root) {
            return isSymmetric(root->left, root->right);
        } else return true;
    }

    bool isSymmetric(TreeNode *root1, TreeNode *root2) {
        if ((!root2 && root1) || (root2 && !root1))
            return false;
        return (root1 == nullptr && root2 == nullptr) ||
               (root1->val == root2->val && isSymmetric(root1->left, root2->right) &&
                isSymmetric(root1->right, root2->left));
    }

    int minNumberOfHours(int initialEnergy, int initialExperience, vector<int> &energy, vector<int> &experience) {
        int res = max(accumulate(energy.begin(), energy.end(), -initialEnergy + 1), 0);
        int curExperience = initialExperience;
        for (auto i: experience) {
            if (curExperience <= i) {
                res += (i + 1 - curExperience);
                curExperience = 1 + i;
            }
            curExperience += i;
        }
        return res;
    }

    bool mergeTriplets(vector<vector<int>> &triplets, vector<int> &target) {
        bool flag[3] = {false, false, false};
        for (auto &i: triplets) {
            if (i[0] > target[0] || i[1] > target[1] || i[2] > target[2]) {
                continue;
            }
            for (auto j = 0; j < 3; ++j)
                flag[j] = flag[j] || (i[j] == target[j]);
        }
        return flag[0] && flag[1] && flag[2];
    }

    double champagneTower(int poured, int query_row, int query_glass) {
        int depth = 0;
        vector<double> glass(query_glass + 1);
        glass[0] = poured;
        double prev;
        double half;
        while (depth++ < query_row) {
            glass[0] = prev = max(0.0, (glass[0] - 1) / 2);
            for (int i = 1; i <= query_glass; ++i) {
                half = max(0.0, (glass[i] - 1) / 2);
                glass[i] = half + prev;
                prev = half;
            }
        }
        return min(1.0, glass[query_glass]);
    }

    long long interchangeableRectangles(vector<vector<int>> &rectangles) {
        unordered_map<double, long long> hm;
        long long res = 0;
        for (auto &i: rectangles) {
            res += hm[((double) i[0]) / i[1]]++;
        }
        return res;
    }

//    We make good use of the condition "n is odd" as follow
//    a1,(a2,a3),(a4,a5).....,
//    making the decoded into pairs.
//    a2^a3 = A[1]
//    a4^a5 = A[3]
//    a6^a7 = A[5]
//    ...
//    so we can have the result of a2^a3^a4...^an.
//    And a1,a2,a3... is a permutation of 1,2,3,4...n
//
//    so we can have
//    a1 = 1^2^3...^n^a2^a2^a3...^an
//
//    Then we can deduct the whole decoded array.
    vector<int> decode(vector<int> &encoded) {
        int n = encoded.size() + 1;
        int temp = 0;
        int last = n;
        while (n--) {
            last ^= n;
        }
        for (int i = 0; i < encoded.size(); i += 2) {
            last ^= encoded[i];
        }
        for (auto i = encoded.rbegin(); i != encoded.rend(); ++i) {
            *i ^= temp;
            temp = *i;
            *i ^= last;
        }
        encoded.push_back(last);
        return encoded;
    }

    int sumNumbers(TreeNode *root) {
        auto dfs = [](auto &&dfs, string &s, int &res, TreeNode *root) -> void {
            if (root != nullptr) {
                s.push_back(root->val + '0');
                if (root->left == nullptr && root->right == nullptr) {
                    res += stoi(s);
                } else {
                    if (root->left != nullptr) {
                        dfs(dfs, s, res, root->left);
                    }
                    if (root->right != nullptr) {
                        dfs(dfs, s, res, root->right);
                    }
                }
                s.pop_back();
            }
        };
        int res = 0;
        string s;
        dfs(dfs, s, res, root);
        return res;
    }

//    The greedy pick won't break anything, so just take as much as possible.
//    For each result value at A[i][j],
//            we greedily take the min(row[i], col[j]).
//    Then we update the row sum and col sum:
//            row[i] -= A[i][j]
//    col[j] -= A[i][j]
    vector<vector<int>> restoreMatrix(vector<int> &rowSum, vector<int> &colSum) {
        vector<vector<int>> res(rowSum.size(), vector<int>(colSum.size(), 0));
        for (auto i = 0; i < rowSum.size(); ++i) {
            for (auto j = 0; j < colSum.size(); ++j) {
                int num = min(rowSum[i], colSum[j]);
                res[i][j] = num;
                rowSum[i] -= num;
                colSum[j] -= num;
            }
        }
        return res;
    }

    vector<string> buildArray(vector<int> &target, int n) {
        string push = "Push";
        string pop = "Pop";
        vector<string> res;
        auto cur = target.begin();
        for (int i = 1; i <= n; ++i) {
            if (cur == target.end()) {
                return res;
            } else {
                res.push_back(push);
                if (*cur != i) {
                    res.push_back(pop);
                } else {
                    ++cur;
                }
            }
        }
        return res;
    }

    string smallestFromLeaf(TreeNode *root) {
        string res, cur;
        auto dfs = [](auto &&dfs, TreeNode *root, string &res, string &cur) -> void {
            if (root != nullptr) {
                cur.push_back('a' + root->val);
                if (root->left || root->right) {
                    dfs(dfs, root->left, res, cur);
                    dfs(dfs, root->right, res, cur);
                } else {
                    string reverse(cur.rbegin(), cur.rend());
                    if (res.empty() || res > reverse) {
                        res = reverse;
                    }
                }
                cur.pop_back();
            }
        };
        dfs(dfs, root, res, cur);
        return res;
    }

    vector<int> findMode(TreeNode *root) {
        vector<int> res;
        vector<int> tree;
        int times = 1;
        int most = 0;
        auto dfs = [](auto &&dfs, vector<int> &tree, TreeNode *root) -> void {
            if (root) {
                dfs(dfs, tree, root->left);
                tree.push_back(root->val);
                dfs(dfs, tree, root->right);
            }
        };
        dfs(dfs, tree, root);
        int prev = tree[0] - 1;
        for (auto i: tree) {
            if (prev == i) {
                ++times;
            } else {
                times = 1;
            }
            if (times > most) {
                most = times;
                res = {i};
            } else if (times == most) {
                res.push_back(i);
            }
            prev = i;
        }
        return res;
    }

    int minOperations(vector<int> &nums) {
        int maxNum = 0;
        int res = 0;
        int cur, temp;
        int plus = 0;
        for (auto &i: nums) {
            plus = 0;
            cur = -1;
            temp = i;
            while (temp >= 1) {
                plus += temp & 1;
                temp >>= 1;
                ++cur;
            }
            res += plus;
            maxNum = max(maxNum, cur);
        }
        return res + maxNum;
    }

    vector<int> memLeak(int memory1, int memory2) {
        int time = 1;
        while (memory1 >= time || memory2 >= time) {
            if (memory1 < time || memory2 > memory1) {
                memory2 -= time;
            } else {
                memory1 -= time;
            }
            ++time;
        }
        return {time, memory1, memory2};
    }

    vector<int> avoidFlood(vector<int> &rains) {
        vector<int> res(rains.size(), -1);
        unordered_map<int, int> fulls;
        set<int> s;
        for (auto i = 0; i < rains.size(); ++i) {
            if (rains[i] == 0) {
                s.insert(i);
                res[i] = 1;
            } else {
                auto pair = fulls.find(rains[i]);
                if (pair == fulls.end()) {
                    fulls[rains[i]] = i;
                } else {
                    auto dry = s.upper_bound(pair->second);
                    if (dry == s.end()) {
                        return {};
                    } else {
                        res[*dry] = rains[i];
                        s.erase(dry);
                        fulls[rains[i]] = i;
                    }
                }
            }
        }
        return res;
    }

    string customSortString(string order, string s) {
        unordered_map<char, int> hm;
        for (auto i = 0; i < order.size(); ++i) {
            hm[order[i]] = i;
        }
        auto f = [&](const char &a, const char &b) {
            return hm[a] < hm[b];
        };
        sort(s.begin(), s.end(), f);
        return s;
    }

    vector<int> getMaximumXor(vector<int> &nums, int maximumBit) {
        int maxNum = (1 << (maximumBit)) - 1;
        for (auto i = nums.begin(); i != nums.end(); ++i) {
            *i ^= (i == begin(nums) ? maxNum : *(i - 1));
        }
        return std::move(vector<int>(nums.rbegin(), nums.rend()));
    }

    bool isCompleteTree(TreeNode *root) {
        queue<TreeNode *> q;
        queue<TreeNode *> temp;
        q.push(root);
        bool flag = true;
        while (!q.empty()) {
            temp = queue<TreeNode *>();
            while (!q.empty()) {
                auto front = q.front();
                if (front == nullptr) {
                    flag = false;
                } else {
                    if (!flag)
                        return false;
                    temp.push(front->left);
                    temp.push(front->right);
                }
                q.pop();
            }
            swap(temp, q);
        }
        return true;
    }

    int maximalNetworkRank(int n, vector<vector<int>> &roads) {
        vector<unordered_set<int>> nexts(n);
        for (auto &i: roads) {
            nexts[i[0]].insert(i[1]);
            nexts[i[1]].insert(i[0]);
        }
        int res = 0;
        for (auto i = nexts.begin(); i != nexts.end(); ++i) {
            for (auto j = i + 1; j != nexts.end(); ++j) {
                res = max(res, (int) (i->size() + j->size() - j->count(i - begin(nexts))));
            }
        }
        return res;
    }

    vector<int> searchRange(vector<int> &nums, int target) {
        auto first = lower_bound(nums.begin(), nums.end(), target);
        auto last = upper_bound(nums.begin(), nums.end(), target);
        if (first == nums.end() || first == last) {
            return {-1, -1};
        } else {
            return {(int) (first - nums.begin()), (int) (last - nums.begin()) - 1};
        }
    }

    vector<int> rearrangeArray(vector<int> &nums) {
        int pos = 0, neg = 1;
        vector<int> res = nums;
        for (auto i = 0; i < nums.size(); i++) {
            if (nums[i] > 0) {
                res[pos] = nums[i];
                pos += 2;
            }
            if (nums[i] < 0) {
                res[neg] = nums[i];
                neg += 2;
            }
        }
        return std::move(res);
    }

    long long smallestNumber(long long num) {
        vector<int> hm(10);
        string res;
        if (num == 0) {
            return 0;
        }
        bool flag = false;
        if (num < 0) {
            flag = true;
            num = -num;
        }
        do {
            hm[num % 10]++;
        } while ((num = num / 10) >= 1);
        if (flag) {
            int i = 9;
            while (!hm[i]) {
                --i;
            }
            --hm[i];
            res.push_back('-');
            res.push_back(i + '0');
            for (int i = 9; i >= 0; --i) {
                res.append(hm[i], i + '0');
            }
        } else {
            int i = 1;
            while (!hm[i]) {
                ++i;
            }
            --hm[i];
            res.push_back(i + '0');
            for (int i = 0; i <= 9; ++i) {
                res.append(hm[i], i + '0');
            }
        }
        return stoll(res);
    }

    int minImpossibleOR(vector<int> &v) {
        int res;
        unordered_set<int> s(v.begin(), v.end());
        while (s.find(res) != s.end())res <<= 1;
        return res;
    }

//    In this example, using auto&& has a reason. auto&& is a universal reference, which can bind to any type of value, including lvalues and rvalues. This is very useful for generic programming and perfect forwarding.
//
//    In the Y-combinator example, we need to pass the lambda function f as the first argument to itself. Using auto&& here ensures that, during the recursive call, f is passed to itself with the correct reference type.
//
//    By using auto&&, we can ensure that the reference to f is correct in the recursive call, whether f is an lvalue or an rvalue. This can reduce unnecessary copying and ensure the efficiency of the code.
//
//    If we use auto& or auto instead of auto&&, it might lead to copying or reference errors during the recursive call. This is because auto& can only bind to lvalues, while auto creates a new object, resulting in unnecessary copying in the recursive call. Therefore, using auto&& is a more suitable choice in this example.

    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
        using iter = vector<int>::iterator;
        auto f = [](auto &&f, iter inorderBegin, iter inorderEnd, iter postorderBegin,
                    iter postorderEnd) -> TreeNode * {
            if (inorderEnd == inorderBegin || postorderBegin == postorderEnd) {
                return nullptr;
            } else {
                auto *root = new TreeNode(*(postorderEnd - 1));
                iter pos = find(inorderBegin, inorderEnd, root->val);
                auto len = (pos - inorderBegin);
                root->left = f(f, inorderBegin, pos, postorderBegin, postorderBegin + len);
                root->right =
                        f(f, pos + 1, inorderEnd, postorderBegin + len, postorderEnd - 1);
                return root;
            }
        };
        return f(f, inorder.begin(), inorder.end(), postorder.begin(), postorder.end());
    }

    int countSubarrays(vector<int> &nums, int k) {
        int res = 0;
        unordered_map<int, int> hm;
        int prefixSum = 0;
        hm[0] = 1;
        bool checked = false;
        for (auto &i: nums) {
            if (i > k) {
                i = 1;
            } else if (i < k) {
                i = -1;
            } else {
                i = 0;
            }
        }
        for (auto i: nums) {
            if (i == 0) {
                checked = true;
            }
            prefixSum += i;
            if (checked) {
                res += hm[prefixSum] + hm[prefixSum - 1];
            } else {
                ++hm[prefixSum];
            }
        }
        return res;
    }

    int minSubarray(vector<int> &nums, int p) {
        int sum = 0;
        int res = -1;
        unordered_map<int, int> hm;
        int remainder = static_cast<int>(accumulate(nums.begin(), nums.end(), (long long) 0) % p);
        if (remainder == 0) {
            return 0;
        }
        hm[0] = -1;
        for (auto i = 0; i < nums.size(); ++i) {
            sum += nums[i];
            sum %= p;
            auto iter = hm.find((sum + p - remainder) % p);
            if (iter != hm.end()) {
                auto len = i - iter->second;
                if ((res == -1 || res > len) && (len != nums.size())) {
                    res = len;
                }
            }
            hm[sum] = i;
        }
        return res;
    }

    int makeStringSorted(string &s) {
        using ll = long long;
        ll res = 0;
        const ll mod = 1e9 + 7;
        vector<int> nums(26, 0);
        vector<long long> fact(s.size() + 1, 1ll);
        for (auto i: s) {
            ++nums[i - 'a'];
        }

        // b^p = (b^(p/2))^2 + b*(p%2==1)
        //     = (b^2)^(p>>1) + b*(p%2==1)
        auto modpow = [&](ll b, ll p) -> ll {
            ll ans = 1;
            do {
                if (p & 1) {
                    ans = (ans * b) % mod;
                }
                b = b * b % mod;
            } while (p >>= 1);
            return ans;
        };
        // to boost factorial
        for (auto i = 1; i <= s.size(); i++) {
            fact[i] = (fact[i - 1] * i) % mod;
        }
        auto l = s.size();
        for (char c: s) {
            l--;
            long long t = 0, rev = 1;
            for (int i = 0; i < 26; i++) {
                if (i < c - 'a')
                    t += nums[i];
                rev = (rev * fact[nums[i]]) % mod;
            }
//            According to Fermat's Little Theorem, for a prime modulus p and an integer a that is not divisible by p,
//            the following holds: a^(p-1) ≡ 1 (mod p).
//            Therefore, the modular multiplicative inverse a^(-1) is a^(p-2)
//            a/b mod m == a*b’mod m b' is b's modular multiplicative inverse
//            the modular division a / b is equivalent to a * (b^(-1)), or a * modpow(b, p-2).
            res += (t * fact[l] % mod) * modpow(rev, mod - 2);
            res %= mod;
            nums[c - 'a']--;
        }
        return res;
    }

    bool checkPalindromeFormation(const string &a, const string &b) {
        auto checkInclusive = [](const string &s) -> bool {
            if (s.empty()) { return false; }
            if (s.size() == 1) { return true; }
            auto i = s.begin();
            auto j = s.end() - 1;
            while (i < j) {
                if (*i++ != *j--) { return false; }
            }
            return true;
        };

        auto f = [&](const string &a, const string &b) -> bool {
            auto i = 0;
            auto j = b.size() - 1;
            while (i < j && i < b.size() && j >= 0 && a[i] == b[j]) {
                ++i, --j;
            }
            if (i > j) {
                return true;
            } else {
                return checkInclusive(a.substr(i, j - i + 1)) || checkInclusive(b.substr(i, j - i + 1));
            }
        };

        return f(a, b) || f(b, a);
    }

    bool canPlaceFlowers(vector<int> &flowerbed, int n) {
        for (auto i = 0; i < flowerbed.size(); ++i) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) &&
                (i == flowerbed.size() - 1 || flowerbed[i + 1] == 0)) {
                --n;
                flowerbed[i] = 1;
            }
            if (n <= 0) {
                return true;
            }
        }
        return false;
    }

    int numDupDigitsAtMostN(int n) {
        vector<int> prev_ans = {0, 0, 9, 261, 4725, 67509, 831429, 9287109, 97654149, 994388229};
        using ll = long long;
        auto pow = [&](ll b, ll p) -> ll {
            ll ans = 1;
            do {
                if (p & 1) {
                    ans = (ans * b);
                }
                b = b * b;
            } while (p >>= 1);
            return ans;
        };

        vector<ll> fact(10 + 1, 1ll);
        for (auto i = 1; i < fact.size(); i++) {
            fact[i] = (fact[i - 1] * i);
        }

        string num = to_string(n);
        unordered_set<char> set(num.begin(), num.end());
        auto len = num.size();
        ll res = (len - set.size() > 0 ? 1 : 0);
        int l = 1;
        bitset<10> flag(0);
        ll unique = 0;
        bool has_smaller_unique = true;
        for (auto i = 0; i < len; ++i, ++l) {
            char c = num[i];
            int number = (c - '0');
            if (i == 0) {
                flag.set(number);
                res += (number - 1) * pow(10, len - l);
                unique += (number - 1) * fact[9] / fact[9 - len + l];
            } else {
                res += (number) * pow(10, len - l);
                if (has_smaller_unique) {
                    int cur_num = number - 1;
                    while (cur_num >= 0 && flag[cur_num]) {
                        --cur_num;
                    }
                    if (cur_num >= 0 && !flag[cur_num]) {
                        int choice = 0;
                        for (int k = 0; k <= cur_num; ++k) {
                            if (!flag[k]) {
                                ++choice;
                            }
                        }
                        int remains = 10 - flag.count() - 1;
                        unique += (choice) * fact[remains] / fact[remains - len + l];
                    }
                    if (flag[number]) {
                        has_smaller_unique = false;
                    }
                    if (!flag[number]) {
                        flag.set(number);
                    } else if (cur_num >= 0 && !flag[cur_num]) {
                        flag.set(cur_num);
                    }
                }
            }
        }
        return static_cast<int>(res - unique + prev_ans[len - 1]);
    }

    int majorityElement(vector<int> &nums) {
        //Boyer-Moore
        int m = nums[0];
        int num = 0;
        for (auto i: nums) {
            if (i == m) {
                ++num;
            } else {
                --num;
                if (num < 0) {
                    m = i;
                    num = 1;
                }
            }
        }
        return m;
    }

    long long zeroFilledSubarray(vector<int> &nums) {
        long long len = 0;
        long long res = 0;
        for (auto i: nums) {
            if (i == 0) {
                ++len;
            } else {
                if (len > 0) {
                    res += len * (len + 1) / 2;
                    len = 0;
                }
            }
        }
        if (len > 0) {
            res += len * (len + 1) / 2;
            len = 0;
        }
        return res;
    }

    string validIPAddress(const string &queryIP) {
        static std::regex ipv4_pattern(
                "^((25[0-5]|2[0-4][0-9]|[1][0-9][0-9]|[1-9][0-9]|[0-9])(\\.)){3}(25[0-5]|2[0-4][0-9]|[1][0-9][0-9]|[1-9][0-9]|[0-9])$");
        static std::regex ipv6_pattern(
                "^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$");
        if (queryIP.empty()) {
            return "Neither";
        }
        if (regex_match(queryIP, ipv4_pattern)) {
            return "IPv4";
        } else if (regex_match(queryIP, ipv6_pattern)) {
            return "IPv6";
        } else {
            return "Neither";
        }
    }

    int calculate(string s) {
        stack<char> ops;
        string r;
        bool flag_p_d = false;
        long long int temp;
        for (auto i: s) {
            if (isdigit(i)) {
                if (!flag_p_d) {
                    flag_p_d = true;
                    temp = i - '0';
                } else {
                    temp *= 10;
                    temp += i - '0';
                }
            } else {
                if (flag_p_d) {
                    flag_p_d = false;
                    r.append(to_string(temp));
                    r.push_back(',');
                    temp = 0;
                }
            }

            if (i == '(') {
                ops.push(i);
            } else if (i == '+' || i == '-') {
                if (ops.empty()) {
                    ops.push(i);
                } else {
                    while (!ops.empty() && ops.top() != '(') {
                        char op = ops.top();
                        ops.pop();
                        r.push_back(op);
                    }
                    ops.push(i);
                }
            } else if (i == ')') {
                while (!ops.empty() && ops.top() != '(') {
                    char op = ops.top();
                    ops.pop();
                    r.push_back(op);
                }
                if (ops.top() == '(')
                    ops.pop();
            }
        }
        if (flag_p_d) {
            flag_p_d = false;
            r.append(to_string(temp));
            r.push_back(',');
            temp = 0;
        }
        while (!ops.empty()) {
            char op = ops.top();
            ops.pop();
            r.push_back(op);
        }
        cout << r << endl;
        stack<int> rnums;
        for (auto i: r) {
            if (isdigit(i)) {
                if (!flag_p_d) {
                    flag_p_d = true;
                    temp = i - '0';
                } else {
                    temp *= 10;
                    temp += i - '0';
                }
            } else {
                if (flag_p_d) {
                    flag_p_d = false;
                    rnums.push(temp);
                    temp = 0;
                }
            }
            if (i == '+') {
                int num1 = rnums.top();
                rnums.pop();
                int num2 = rnums.top();
                rnums.pop();
                rnums.push(num1 + num2);
            } else if (i == '-') {
                int num1 = rnums.top();
                rnums.pop();
                int num2 = rnums.top();
                rnums.pop();
                rnums.push(num2 - num1);
            }
        }
        return rnums.top();
    }

    int diameterOfBinaryTree(TreeNode *root) {
        int res = 0;
        auto diameterOfBinaryTreeDFS = [&](auto &&diameterOfBinaryTreeDFS, TreeNode *root) -> int {
            if (root == nullptr) {
                return 0;
            }
            int left = diameterOfBinaryTreeDFS(diameterOfBinaryTreeDFS, root->left);
            int right = diameterOfBinaryTreeDFS(diameterOfBinaryTreeDFS, root->right);
            res < left + right ? (res = left + right) : res;
            return (left > right ? left : right) + 1;
        };
        diameterOfBinaryTreeDFS(diameterOfBinaryTreeDFS, root);
        return res;
    }

    vector<string> getFolderNames(const vector<string> &names) {
        vector<string> res;
        res.reserve(names.size());
        unordered_map<string, int> hm;
        for (const auto &i: names) {
            auto iter = hm.find(i);
            if (iter == hm.end()) {
                res.push_back(i);
                ++iter->second;
            } else {
                int k = iter->second;
                string temp = i + '(' + to_string(k) + ')';
                auto temp_iter = hm.find(temp);
                while (temp_iter != hm.end()) {
                    ++k;
                    temp = i + '(' + to_string(k) + ')';
                    temp_iter = hm.find(temp);
                }
                iter->second = k + 1;
                hm[temp] = 1;
                res.push_back(temp);
            }
        }
        return res;
    }

    bool halvesAreAlike(const string &s) {
        int size = s.size();
        int a = 0;
        int b = 0;
        for (int i = 0; i < size / 2; ++i) {
            if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' || s[i] == 'E' ||
                s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
                ++a;
            }
        }
        for (int i = size / 2; i < size; ++i) {
            if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' || s[i] == 'E' ||
                s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
                ++b;
            }
        }
        return a == b;
    }

    Node *construct(vector<vector<int>> &grid) {
        auto all_equals = [&](vector<vector<int>> &grid, int startx, int starty, int len) -> int {
            int temp = grid[startx][starty];
            for (int i = 0; i < len; ++i) {
                for (int j = 0; j < len; ++j) {
                    if (grid[startx + i][starty + j] == temp) {
                        continue;
                    } else {
                        return 0;
                    }
                }
            }
            return temp == 1 ? 1 : -1;
        };

        auto dfs = [&](auto &&dfs, vector<vector<int>> &grid, int start_x, int start_y, int len) -> Node * {
            if (len == 1) {
                return new Node((grid[start_x][start_y] == 1), true);
            } else {
                int value = all_equals(grid, start_x, start_y, len);
                if (value == 0) {
                    return new Node((grid[start_x][start_y] == 1), false,
                                    dfs(dfs, grid, start_x, start_y, len / 2),
                                    dfs(dfs, grid, start_x, start_y + len / 2, len / 2),
                                    dfs(dfs, grid, start_x + len / 2, start_y, len / 2),
                                    dfs(dfs, grid, start_x + len / 2, start_y + len / 2, len / 2));
                } else {
                    return new Node((value == 1), true);
                }
            }
        };
        return dfs(dfs, grid, 0, 0, grid.size());
    }

    vector<vector<int>> outerTrees(vector<vector<int>> &trees) {
        auto crossProduct = [](vector<int> p1, vector<int> p2, vector<int> p3) {
            int a = p2[0] - p1[0];
            int b = p2[1] - p1[1];
            int c = p3[0] - p1[0];
            int d = p3[1] - p1[1];
            return a * d - b * c;
        };

        auto constructHalfHull = [&](vector<vector<int>> &trees) -> vector<vector<int>> {
            vector<vector<int>> q;
            for (int i = 0; i < trees.size(); ++i) {
                while (q.size() >= 2 && crossProduct(*(q.end() - 2), *(q.end() - 1), trees[i]) > 0) {
                    q.pop_back();
                }
                q.push_back(trees[i]);
            }
            return q;
        };

        if (trees.size() <= 3) {
            return trees;
        }
        sort(trees.begin(), trees.end());

        auto m = constructHalfHull(trees);
        vector<vector<int>> temp;
        for (auto i = trees.rbegin(); i != trees.rend(); ++i) {
            temp.push_back(*i);
        }
        auto n = constructHalfHull(temp);
        set<vector<int>> r;
        for (auto i: m) {
            r.insert(i);
        }
        for (auto i: n) {
            r.insert(i);
        }
        vector<vector<int>> q;
        for (auto i: r) {
            q.push_back(i);
        }

        return q;
    }


    vector<vector<int>> threeSum(vector<int> &nums) {
        static const long long int base = 200000 + 1;
        static const long long int sqbase = base * base;
        static const long long int bias = 100000;

        auto bisect_left = [&](vector<int> &nums, int x, int a, int b) {
            int mid = (a + b) / 2;
            while (mid < b && a < b) {
                if (nums[mid] == x) {
                    return mid;
                } else if (nums[mid] > x) {
                    b = mid - 1;
                    mid = (a + b) / 2;
                } else {
                    a = mid + 1;
                    mid = (a + b) / 2;
                }
            }
            return mid;
        };

        int n = nums.size();
        unordered_set<long long int> res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1; j < n; ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                int dif = 0 - nums[i] - nums[j];
                int idx = 0;
                while (idx < n) {
                    idx = bisect_left(nums, dif, idx + 1, n);
                    if (idx == n || nums[idx] > dif)
                        break;
                    else if (i != idx && idx != j && nums[idx] == dif) {
                        vector<int> l = {i, j, idx};
                        sort(l.begin(), l.end());
                        res.insert((l[0] + bias) * sqbase + (l[1] + bias) * base + l[2] + bias);
                    }
                }

            }
        }
        vector<vector<int>> realres;
        for (auto i: res) {
            int a = i / sqbase;
            int b = (i - a * sqbase) / base;
            int c = (i - a * sqbase - b * base);
            vector<int> temp = {a, b, c};
            realres.push_back(temp);
        }
        return realres;

    }

    bool exist(vector<vector<char>> &board, string word) {
        const int dx[4] = {0, 0, 1, -1};
        const int dy[4] = {1, -1, 0, 0};

        auto next_procedure = [&](auto &&next_procedure, vector<vector<char>> &board, string &word, int index, int x,
                                  int y, vector<vector<bool>> &checks) -> bool {
            ++index;
            if (index == word.size())
                return true;
            for (int i = 0; i < 4; ++i) {
                if (x + dx[i] >= 0 && x + dx[i] < board.size() && y + dy[i] >= 0 && y + dy[i] < board.front().size()) {
                    if (!checks[x + dx[i]][y + dy[i]] && board[x + dx[i]][y + dy[i]] == word[index]) {
                        checks[x + dx[i]][y + dy[i]] = true;
                        if (next_procedure(next_procedure, board, word, index, x + dx[i], y + dy[i], checks)) {
                            return true;
                        }
                        checks[x + dx[i]][y + dy[i]] = false;
                    }
                }
            }
            return false;
        };

        vector<bool> temp(10, false);
        vector<vector<bool>> checks(10, temp);
        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board.front().size(); ++j) {
                if (board[i][j] == word.front()) {
                    checks[i][j] = true;
                    if (next_procedure(next_procedure, board, word, 0, i, j, checks)) {
                        return true;
                    }
                    checks[i][j] = false;
                }
            }
        }
        return false;
    }

    int findFinalValue(vector<int> &nums, int original) {
        bool flag = true;
        while (flag) {
            flag = false;
            for (auto i = 0; i < nums.size(); ++i) {
                if (nums[i] == original) {
                    original *= 2;
                    flag = true;
                }
            }
        }
        return original;
    }

    int minScore(int n, vector<vector<int>> &roads) {
        vector<int> parent(n + 1, 0);
        vector<unsigned long long> len(n + 1, 1);
        for (auto i = 0; i < n + 1; ++i) {
            parent[i] = i;
        }
        auto find = [&](int node) -> int {
            int p = node;
            while (p != parent[p]) {
                p = parent[p];
            }
            return p;
        };

        auto update = [&](int x, int p) {
            int temp;
            while (x != (temp = parent[x])) {
                parent[x] = p;
                x = temp;
            }
        };

        auto uni = [&](int a, int b) -> void {
            int ap = find(a);
            int bp = find(b);
            if (len[ap] > len[bp]) {
                update(b, ap);
                parent[bp] = ap;
                len[ap] += len[bp];
            } else {
                update(a, bp);
                parent[ap] = bp;
                len[bp] += len[ap];
            }
        };

        for (auto &i: roads) {
            uni(i[0], i[1]);
        }
        unordered_map<int, int> hm;
        for (auto &i: roads) {
            int ap = find(i[0]);
            int bp = find(i[1]);
            if (hm.find(ap) == hm.end()) {
                hm[ap] = i[2];
            } else {
                hm[ap] = min(hm[ap], i[2]);
            }
            if (hm.find(bp) == hm.end()) {
                hm[bp] = i[2];
            } else {
                hm[bp] = min(hm[bp], i[2]);
            }
        }

        return hm[find(1)];
    }

    int findKthNumber(int n, int k) {
        auto count_nodes_with_prefix = [](long long n, long long prefix) {
            long long count = 0;
            long long current = prefix;
            long long next = prefix + 1;
            while (current <= n) {
                count += std::min(n + 1, next) - current;
                current *= 10;
                next *= 10;
            }
            return count;
        };
        int prefix = 1;
        while (k > 1) {
            long long count = count_nodes_with_prefix(n, prefix);
            if (k <= count) {
                prefix *= 10;
                k -= 1;
            } else {
                prefix += 1;
                k -= count;
            }
        }
        return prefix;
    }

    int makeConnected(int n, vector<vector<int>> &connections) {
        vector<int> parent(n + 1, 0);
        vector<unsigned long long> len(n + 1, 1);
        for (auto i = 0; i < n + 1; ++i) {
            parent[i] = i;
        }
        auto find = [&](int node) -> int {
            int p = node;
            while (p != parent[p]) {
                p = parent[p];
            }
            return p;
        };

        auto update = [&](int x, int p) {
            int temp;
            while (x != (temp = parent[x])) {
                parent[x] = p;
                x = temp;
            }
        };

        auto uni = [&](int a, int b) -> void {
            int ap = find(a);
            int bp = find(b);
            if (len[ap] > len[bp]) {
                update(b, ap);
                parent[bp] = ap;
                len[ap] += len[bp];
            } else {
                update(a, bp);
                parent[ap] = bp;
                len[bp] += len[ap];
            }
        };

        int remains = 0;
        for (auto &i: connections) {
            int ap = find(i[0]);
            int bp = find(i[1]);
            if (ap == bp) {
                ++remains;
            } else {
                uni(i[0], i[1]);
            }
        }

        unordered_set<int> subGraphs;
        for (auto i = 0; i < n; ++i) {
            subGraphs.insert(find(i));
        }
        int subGraphSize = subGraphs.size();
        if (subGraphSize > remains + 1) {
            return -1;
        } else if (subGraphSize == 1) {
            return 0;
        } else {
            return subGraphSize - 1;
        }
    }

    vector<bool> checkArithmeticSubarrays(vector<int> &nums, vector<int> &l, vector<int> &r) {
        vector<bool> res(l.size(), true);
        for (auto i = 0; i < l.size(); ++i) {
            auto [p_min, p_max] = minmax_element(begin(nums) + l[i], begin(nums) + r[i] + 1);
            int len = r[i] - l[i] + 1, d = (*p_max - *p_min) / (len - 1);
            if (*p_max == *p_min)
                continue;
            else if ((*p_max - *p_min) % (len - 1)) {
                res[i] = false;
                continue;
            } else {
                vector<bool> n(len);
                int j;
                for (j = l[i]; j <= r[i]; ++j) {
                    if ((nums[j] - *p_min) % d || n[(nums[j] - *p_min) / d]) {
                        break;
                    }
                    n[(nums[j] - *p_min) / d] = true;
                }
                res[i] = (j > r[i]);
            }
        }
        return std::move(res);
    }

    double myPow(double x, long long n) {
        double res = 1;
        if (n == 0) {
            return res;
        } else if (n < 0) {
            n = -n;
            x = 1.0 / x;
        }
        do {
            if (n & 1) {
                res *= x;
            }
            x *= x;
        } while (n >>= 1);
        return res;
    }

    int minReorder(int n, vector<vector<int>> &connections) {
        int res = 0;
        vector<vector<int>> to(n, vector<int>());
        vector<vector<int>> from(n, vector<int>());
        vector<bool> unvisited(n, true);
        unvisited[0] = false;
        for (auto &i: connections) {
            to[i[0]].push_back(i[1]);
            from[i[1]].push_back(i[0]);
        }
        queue<int> q;
        q.push(0);
        while (!q.empty()) {
            int front = q.front();
            q.pop();
            for (auto i: from[front]) {
                if (unvisited[i]) {
                    unvisited[i] = false;
                    q.push(i);
                }
            }
            for (auto i: to[front]) {
                if (unvisited[i]) {
                    unvisited[i] = false;
                    q.push(i);
                    ++res;
                }
            }
        }
        return res;
    }

    long long countPairs(int n, vector<vector<int>> &edges) {
        vector<int> parent(n + 1, 0);
        vector<unsigned long long> len(n + 1, 1);
        for (auto i = 0; i < n + 1; ++i) {
            parent[i] = i;
        }
        auto find = [&](int node) -> int {
            int p = node;
            while (p != parent[p]) {
                p = parent[p];
            }
            return p;
        };

        auto update = [&](int x, int p) {
            int temp;
            while (x != (temp = parent[x])) {
                parent[x] = p;
                x = temp;
            }
        };

        auto uni = [&](int a, int b) -> void {
            int ap = find(a);
            int bp = find(b);
            if (ap == bp) {
                return;
            }
            if (len[ap] > len[bp]) {
                update(b, ap);
                parent[bp] = ap;
                len[ap] += len[bp];
            } else {
                update(a, bp);
                parent[ap] = bp;
                len[bp] += len[ap];
            }
        };
        for (auto &i: edges) {
            uni(i[0], i[1]);
        }
        unordered_map<int, long long> subGraphs;
        for (auto i = 0; i < n; ++i) {
            subGraphs[find(i)] += 1;
        }
        long long res = 0;
        for (auto &i: subGraphs) {
            res += i.second * (n - i.second);
        }
        return res / 2;
    }

    int longestCycle(vector<int> &edges) {
        auto n = edges.size();
        int res = -1;
        unordered_map<int, int> cache;
        for (int i = 0; i < n; ++i) {
            if (cache.count(i)) {
                continue;
            }
            int slow = i;
            int len = -1;
            int fast = i;
            vector<int> path;
            do {
                slow = edges[slow];
                if (cache.count(slow)) {
                    break;
                }
                path.push_back(slow);
                if (slow == -1)
                    break;
                if (edges[fast] == -1)
                    break;
                fast = edges[edges[fast]];
                if (fast == -1)
                    break;
                if (slow == fast) {
                    len = 0;
                    int encounter = i;
                    while (encounter != slow) {
                        encounter = edges[encounter];
                        slow = edges[slow];
                        path.push_back(slow);
                    }
                    int temp = encounter;
                    do {
                        ++len;
                        temp = edges[temp];
                    } while (temp != encounter);
                    res = max(res, len);
                    break;
                }
            } while (true);
            for (auto i: path) {
                cache[i] = len;
            }
        }
        return res;
    }

    int minPathSum(vector<vector<int>> &grid) {
        for (int i = 1; i < grid[0].size(); ++i) {
            grid[0][i] += grid[0][i - 1];
        }
        for (int i = 1; i < grid.size(); ++i) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int i = 1; i < grid.size(); ++i) {
            for (int j = 1; j < grid[0].size(); ++j) {
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[grid.size() - 1][grid[0].size() - 1];
    }

    int countSubstrings(const string &s, const string &t) {
        int res = 0;
        vector<vector<int>> right(s.size(), vector<int>(t.size())), left(s.size(), vector<int>(t.size()));
        for (int i = 1; i < s.size(); ++i) {
            for (int j = 1; j < t.size(); ++j) {
                if (s[i - 1] == t[j - 1])
                    left[i][j] = 1 + left[i - 1][j - 1];
            }
        }
        for (int i = s.size() - 2; i >= 0; --i) {
            for (int j = t.size() - 2; j >= 0; --j) {
                if (s[i + 1] == t[j + 1])
                    right[i][j] = 1 + right[i + 1][j + 1];
            }
        }
        for (int i = 0; i < s.size(); ++i) {
            for (int j = 0; j < t.size(); ++j) {
                if (s[i] != t[j]) {
                    res += (left[i][j] + 1) * (right[i][j] + 1);
                }
            }
        }
        return res;
    }

    int mincostTickets(vector<int> &days, vector<int> &costs) {
        int dp[366] = {0};
        bool travelDays[366] = {false};
        for (int day: days) {
            travelDays[day] = true;
        }
        for (int i = 1; i <= 365; i++) {
            if (!travelDays[i]) {
                dp[i] = dp[i - 1];
            } else {
                dp[i] = min(dp[i - 1] + costs[0], min(dp[max(0, i - 7)] + costs[1], dp[max(0, i - 30)] + costs[2]));
            }
        }
        return dp[365];
    }

    long long maximumSubarraySum(vector<int> &nums, int k) {
        long long res = 0;
        long long sum = 0;
        int cur = 0;
        unordered_set<int> selected;
        for (int i = 0; i < nums.size(); ++i) {
            if (selected.find(nums[i]) != selected.end()) {
                do {
                    selected.erase(nums[cur]);
                    sum -= nums[cur];
                } while (nums[cur++] != nums[i]);
            }
            selected.insert(nums[i]);
            sum += nums[i];
            if (selected.size() == k) {
                res = max(res, sum);
                selected.erase(nums[cur]);
                sum -= nums[cur++];
            }
        }
        return res;
    }

    int maxSatisfaction(vector<int> &satisfaction) {
        sort(satisfaction.rbegin(), satisfaction.rend());
        int sum = 0;
        int res = 0;
        int i = 0;
        while (i < satisfaction.size() && sum + satisfaction[i] > 0) {
            sum += satisfaction[i];
            res += sum;
            ++i;
        }
        return res;
    }

    bool checkOverlap(int radius, int xCenter, int yCenter, int x1, int y1, int x2, int y2) {
        if (xCenter < x1) {
            if (yCenter < y1) {
                return (x1 - xCenter) * (x1 - xCenter) + (y1 - yCenter) * (y1 - yCenter) <= radius * radius;
            } else if (yCenter > y2) {
                return (x1 - xCenter) * (x1 - xCenter) + (y2 - yCenter) * (y2 - yCenter) <= radius * radius;
            } else {
                return x1 - xCenter <= radius;
            }
        } else if (xCenter > x2) {
            if (yCenter < y1) {
                return (x2 - xCenter) * (x2 - xCenter) + (y1 - yCenter) * (y1 - yCenter) <= radius * radius;
            } else if (yCenter > y2) {
                return (x2 - xCenter) * (x2 - xCenter) + (y2 - yCenter) * (y2 - yCenter) <= radius * radius;
            } else {
                return xCenter - x2 <= radius;
            }
        } else {
            return abs(yCenter - y1) <= radius || abs(yCenter - y2) <= radius || (yCenter <= y2 && yCenter >= y1);
        }
    }

    int countVowelStrings(int n) {
//        int res = 0;
//        auto f = [&](auto &&f, int depth, int last) -> void {
//            if (depth == n) {
//                res += 5 - last;
//            } else {
//                for (int i = last; i < 5; ++i)
//                    f(f, depth + 1, i);
//            }
//        };
//        f(f, 0, 0);
//        return res;
        int res[50] = {5, 15, 35, 70, 126, 210, 330, 495, 715, 1001, 1365, 1820, 2380, 3060, 3876, 4845, 5985, 7315,
                       8855, 10626, 12650, 14950, 17550, 20475, 23751, 27405, 31465, 35960, 40920, 46376, 52360, 58905,
                       66045, 73815, 82251, 91390, 101270, 111930, 123410, 135751, 148995, 163185, 178365, 194580,
                       211876, 230300, 249900, 270725, 292825, 316251};
        return res[n - 1];
    }

    int numOfMinutes(int n, int headID, vector<int> &manager, vector<int> &informTime) {
        vector<vector<int>> graph(n, vector<int>());
        for (auto i = 0; i < manager.size(); ++i) {
            if (manager[i] != -1)
                graph[manager[i]].push_back(i);
        }
        int res = 0;
        auto f = [&](auto &&f, int prev, int manager) -> void {
            res = max(res, (prev += informTime[manager]));
            for (auto i: graph[manager]) {
                if (!graph[i].empty()) {
                    f(f, prev, i);
                }
            }
        };
        f(f, 0, headID);
        return res;
    }

    int subarrayBitwiseORs(vector<int> &arr) {
        unordered_set<int> res;
        unordered_set<int> prev;
        for (auto i = 0; i < arr.size(); ++i) {
            prev.insert(0);
            unordered_set<int> current;
            for (auto prevComputed: prev) {
                current.insert(prevComputed | arr[i]);
                res.insert(prevComputed | arr[i]);
            }
            prev.swap(current);
        }
        return res.size();
    }

    bool isScramble(string s1, string s2) {
        int n = s1.length();
        // Initialize a 3D table to store the results of all possible substrings of the two strings
        vector<vector<vector<bool>>> dp(n + 1, vector<vector<bool>>(n, vector<bool>(n)));

        // Initialize the table for substrings of length 1
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dp[1][i][j] = s1[i] == s2[j];
            }
        }

        // Fill the table for substrings of length 2 to n
        for (int length = 2; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                for (int j = 0; j <= n - length; j++) {
                    // Iterate over all possible lengths of the first substring
                    for (int newLength = 1; newLength < length; newLength++) {
                        // Check if the two possible splits of the substrings are scrambled versions of each other
                        vector<bool> &dp1 = dp[newLength][i];
                        vector<bool> &dp2 = dp[length - newLength][i + newLength];
                        dp[length][i][j] = dp[length][i][j] || (dp1[j] && dp2[j + newLength]);
                        dp[length][i][j] = dp[length][i][j] || (dp1[j + length - newLength] && dp2[j]);
                    }
                }
            }
        }

        // Return whether the entire strings s1 and s2 are scrambled versions of each other
        return dp[n][0][0];
    }

    int maxWidthOfVerticalArea(vector<vector<int>> &points) {
        auto f = [](vector<int> &a, vector<int> &b) -> bool {
            return a[0] < b[0];
        };
        sort(points.begin(), points.end(), f);
        int res = 0;
        for (int i = 1; i < points.size(); ++i) {
            res = max(res, points[i][0] - points[i - 1][0]);
        }
        return res;
    }

    bool hasAllCodes(const string &s, int k) {
        vector<bool> flag(1 << k, false);
        for (int i = 0; i + k <= s.size(); ++i) {
            flag[stoi(s.substr(i, k), 0, 2)] = true;
        }
        return std::all_of(flag.begin(), flag.end(), [&](const auto &item) {
            return item;
        });
    }

    int numTimesAllBlue(vector<int> &flips) {
        int max_temp = 0;
        int res = 0;
        for (int i = 0; i < flips.size(); ++i) {
            max_temp = max(max_temp, flips[i]);
            if (max_temp == i + 1) {
                ++res;
            }
        }
        return res;
    }

    int arithmeticTriplets(vector<int> &nums, int diff) {
        int res = 0;
        unordered_map<int, int> left;
        unordered_map<int, int> right;
        for (auto i = 0; i < nums.size(); ++i) {
            for (auto j = i + 1; j < nums.size() && nums[j] - nums[i] <= diff; ++j) {
                if (nums[j] - nums[i] == diff) {
                    res += left[j] + right[i];
                    ++left[i];
                    ++right[j];
                }
            }
        }
        return res;
    }

    int search(vector<int> &nums, int target) {
        int low = 0;
        int high = nums.size() - 1;
        int mid;
        while (low <= high) {
            mid = (low + high) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }

    int ways(vector<string> &pizza, int k) {
        int m = pizza.size(), n = pizza[0].size();
        vector<vector<vector<int>>> dp(k, vector<vector<int>>(m, vector<int>(n, -1)));
        vector<vector<int>> preSum(m + 1, vector<int>(n + 1, 0));
        for (int r = m - 1; r >= 0; r--)
            for (int c = n - 1; c >= 0; c--)
                preSum[r][c] = preSum[r][c + 1] + preSum[r + 1][c] - preSum[r + 1][c + 1] + (pizza[r][c] == 'A');

        auto dfs = [&](auto &&dfs, int temp_k, int r, int c) -> int {
            if (preSum[r][c] == 0) return 0;
            if (temp_k == 0) return 1;
            if (dp[temp_k][r][c] != -1) return dp[temp_k][r][c];
            int ans = 0;
            for (int nr = r + 1; nr < m; nr++)
                if (preSum[r][c] - preSum[nr][c] > 0)
                    ans = (ans + dfs(dfs, temp_k - 1, nr, c)) % 1000000007;
            for (int nc = c + 1; nc < n; nc++)
                if (preSum[r][c] - preSum[r][nc] > 0)
                    ans = (ans + dfs(dfs, temp_k - 1, r, nc)) % 1000000007;

            return dp[temp_k][r][c] = ans;
        };
        return dfs(dfs, k - 1, 0, 0);
    }

    vector<int> productQueries(int n, vector<vector<int>> &queries) {
        using ll = long long int;
        const int MOD = 1e9 + 7;
        auto modpow = [&](ll b, ll p) -> ll {
            ll ans = 1;
            do {
                if (p & 1) {
                    ans = (ans * b) % MOD;
                }
                b = b * b % MOD;
            } while (p >>= 1);
            return ans;
        };
        vector<int> nums;
        bitset<32> n_bit(n);
        for (int i = 0; i <= 31; ++i) {
            if (n_bit[i]) {
                nums.push_back(1 << i);
            }
        }
        vector<ll> preProduct(nums.size());
        vector<ll> ModPowRes(nums.size());
        preProduct[0] = nums[0];
        ModPowRes[0] = modpow(preProduct[0], MOD - 2);
        for (int i = 1; i < nums.size(); ++i) {
            preProduct[i] = preProduct[i - 1] * (ll) nums[i] % MOD;
            ModPowRes[i] = modpow(preProduct[i], MOD - 2);
        }
        vector<int> res;
        for (auto &i: queries) {
            res.push_back(preProduct[i[1]] % MOD * (i[0] == 0 ? 1 : ModPowRes[i[0] - 1]) % MOD);
        }
        return res;
    }

    long long maxMatrixSum(vector<vector<int>> &matrix) {
        long long res = 0;
        int min_abs = INT32_MAX;
        bool flag = false;
        for (auto &i: matrix) {
            for (auto j: i) {
                res += abs(j);
                min_abs = min(min_abs, abs(j));
                if (j < 0) {
                    flag = !flag;
                }
            }
        }
        if (flag) {
            res -= 2 * min_abs;
        }
        return res;
    }

    ListNode *swapPairs(ListNode *head) {
        auto swap = [](ListNode *a) -> void {
            auto c = a->next->next;
            auto b = a->next;
            a->next = c;
            b->next = c->next;
            c->next = b;
        };
        auto newHead = new ListNode(0);
        newHead->next = head;
        auto i = newHead;
        while (i->next && i->next->next) {
            swap(i);
            i = i->next->next;
        }
        return newHead->next;
    }

//    int countPalindromicSubsequence(const string &s) {
//        vector<vector<int>> count(s.size(), vector<int>('z' - 'a' + 1, 0));
//        vector<int> start('z' - 'a' + 1, -1);
//        vector<int> end('z' - 'a' + 1, -1);
//        int res = 0;
//        for (int i = 0; i < s.size(); ++i) {
//            if (i >= 1) count[i] = count[i - 1];
//            ++count[i][s[i] - 'a'];
//            if (start[s[i] - 'a'] == -1) {
//                start[s[i] - 'a'] = i;
//            }
//            end[s[i] - 'a'] = i;
//        }
//        for (auto i = 0; i < 'z' - 'a' + 1; ++i) {
//            if (end[i] == start[i]) {
//                continue;
//            } else {
//                for (auto j = 0; j < 'z' - 'a' + 1; ++j) {
//                    res += (count[end[i] - 1][j] - count[start[i]][j] > 0);
//                }
//            }
//        }
//        return res;
//    }

    int countPalindromicSubsequence(const string &s1) {
        int res = 0;
        for (char i = 'a'; i <= 'z'; i++) {
            int index1 = s1.find(i);
            if (index1 == -1) continue;
            for (char j = 'a'; j <= 'z'; j++) {
                int index2 = s1.find(j, index1 + 1);
                if (index2 == -1) continue;
                if (s1.find(i, index2 + 1) != -1) res++;
            }
        }
        return res;
    }

    vector<int> successfulPairs(vector<int> &spells, vector<int> &potions, long long success) {
        sort(potions.rbegin(), potions.rend());
        vector<int> res(spells.size());
        for (int i = 0; i < spells.size(); ++i) {
            int high = potions.size() - 1;
            int low = 0;
            int mid;
            while (low <= high) {
                mid = (high + low) / 2;
                if (static_cast<long long > (potions[mid]) * spells[i] >= success) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            res[i] = low;
        }
        return res;
    }

    int numRescueBoats(vector<int> &people, int limit) {
        int res = 0;
        sort(people.begin(), people.end());
        int left = 0, right = people.size() - 1;
        while (left <= right) {
            if (people[left] + people[right] <= limit) {
                left++;
            }
            right--;
            res++;
        }
        return res;
    }

    ListNode *partition(ListNode *head, int x) {
        ListNode *res = new ListNode(0);
        ListNode *res_pt = res;
        ListNode *bigger = new ListNode(0);
        ListNode *bigger_pt = bigger;
        ListNode *pt = head;
        while (pt) {
            if (pt->val >= x) {
                bigger_pt->next = pt;
                bigger_pt = pt;
            } else {
                res_pt->next = pt;
                res_pt = pt;
            }
            pt = pt->next;
            bigger_pt->next = nullptr;
            res_pt->next = nullptr;
        }
        bigger = bigger->next;
        while (bigger) {
            res_pt->next = bigger;
            res_pt = res_pt->next;
            bigger = bigger->next;
        }
        return res->next;
    }

    vector<int> deckRevealedIncreasing(vector<int> &deck) {
        sort(deck.begin(), deck.end());
        if (deck.size() <= 2) {
            return deck;
        }
        deque<int> ans;
        ans.push_front(deck.back());
        deck.pop_back();
        while (!deck.empty()) {
            int temp = ans.back();
            ans.pop_back();
            ans.push_front(temp);
            ans.push_front(deck.back());
            deck.pop_back();
        }
        return std::move(vector<int>(ans.begin(), ans.end()));
    }

    int partitionString(const string &s) {
        int res = 1;
        vector<bool> exist(26, false);
        for (auto i: s) {
            if (exist[i - 'a']) {
                ++res;
                std::fill(exist.begin(), exist.end(), false);
            }
            exist[i - 'a'] = true;
        }
        return res;
    }

    int mergeStones(vector<int> &stones, int k) {
        int n = stones.size();
        if ((n - k) % (k - 1)) {
            return -1;
        }
        vector<int> prefixSum(n, 0);
        prefixSum[0] = stones[0];
        for (int i = 1; i < stones.size(); i++) {
            prefixSum[i] = prefixSum[i - 1] + stones[i];
        }

        vector<vector<vector<int>>> dp(50, vector<vector<int>>(50, vector<int>(50, -1)));
        auto minCost = [&](auto &&minCost, int i, int j, int piles) -> int {
            if (i == j && piles == 1)
                return 0;
            if (i == j)
                return INT_MAX / 4;
            if (dp[i][j][piles] != -1)
                return dp[i][j][piles];
            if (piles == 1) {
                return dp[i][j][piles] = minCost(minCost, i, j, k) +
                                         (i == 0 ? prefixSum[j] : prefixSum[j] - prefixSum[i - 1]);

            } else {
                int cost = INT_MAX / 4;
                for (int t = i; t < j; t++) {
                    cost = min(cost,
                               minCost(minCost, i, t, 1) +
                               minCost(minCost, t + 1, j, piles - 1));
                }
                return dp[i][j][piles] = cost;
            }
        };
        return minCost(minCost, 0, n - 1, 1);
    }

    int minimizeArrayValue(vector<int> &nums) {
        vector<long long int> prefix(nums.size());
        prefix[0] = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            prefix[i] = prefix[i - 1] + nums[i];
        }
        long long int res = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] > res) {
                if ((i + 1) * res >= prefix[i]) {
                    continue;
                } else {
                    res = (prefix[i] + i) / (i + 1);
                }
            }
        }
        return res;
    }

    int commonFactors(int a, int b) {
        int upperBound = min(a, b) + 1;
        int res = 0;
        while (--upperBound) {
            if (a % upperBound || b % upperBound) {
                continue;
            } else {
                ++res;
            }
        }
        return res;
    }

    int closedIsland(vector<vector<int>> &grid) {
        int res = 0;
        int n = grid.size();
        int m = grid.begin()->size();
        int xy[5] = {0, 1, 0, -1, 0};
        vector<vector<bool>> visited(n, vector<bool>(m, false));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                bool flag = true;
                if (grid[i][j] == 0 && !visited[i][j]) {
                    stack<pair<int, int>> q;
                    q.push(pair<int, int>(i, j));
                    visited[i][j] = true;
                    while (!q.empty()) {
                        auto [x, y] = q.top();
                        q.pop();
                        if (x == 0 || x == n - 1 || y == 0 || y == m - 1) {
                            flag = false;
                        }
                        for (int k = 0; k < 4; ++k) {
                            int xx = x + xy[k];
                            int yy = y + xy[k + 1];
                            if (xx >= 0 && yy >= 0 && xx < n && yy < m && grid[xx][yy] == 0 && !visited[xx][yy]) {
                                q.push(pair<int, int>(xx, yy));
                                visited[xx][yy] = true;
                            }
                        }
                    }
                    if (flag) {
                        ++res;
                    }
                }
            }
        }
        return res;
    }

    string baseNeg2(int n) {
        for (int i = 1; i < 31; i += 2) {
            if (1 << i & n) n += 1 << (i + 1);
        }
        string res = std::bitset<32>(n).to_string();
        return n ? std::move(res.substr(res.find('1'))) : "0";
    }

    int numEnclaves(vector<vector<int>> &grid) {
        int res = 0;
        int n = grid.size();
        int m = grid.begin()->size();
        int xy[5] = {0, 1, 0, -1, 0};
        vector<vector<bool>> visited(n, vector<bool>(m, false));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                bool flag = true;
                if (grid[i][j] == 1 && !visited[i][j]) {
                    int size = 0;
                    stack<pair<int, int>> q;
                    q.push(pair<int, int>(i, j));
                    visited[i][j] = true;
                    while (!q.empty()) {
                        auto [x, y] = q.top();
                        ++size;
                        q.pop();
                        if (x == 0 || x == n - 1 || y == 0 || y == m - 1) {
                            flag = false;
                        }
                        for (int k = 0; k < 4; ++k) {
                            int xx = x + xy[k];
                            int yy = y + xy[k + 1];
                            if (xx >= 0 && yy >= 0 && xx < n && yy < m && grid[xx][yy] == 1 && !visited[xx][yy]) {
                                q.push(pair<int, int>(xx, yy));
                                visited[xx][yy] = true;
                            }
                        }
                    }
                    if (flag) {
                        res += size;
                    }
                }
            }
        }
        return res;
    }

    vector<int> smallestSufficientTeam(vector<string> &req_skills, vector<vector<string>> &people) {
//        unordered_map<string, int> to_int;
//        vector<vector<int>> people_int(people.size());
//        for (int index = 0; auto &i: req_skills) {
//            to_int[i] = index++;
//        }
//        int nums_skills = req_skills.size();
//        vector<vector<int>> who_has_skill(nums_skills, vector<int>());
//        for (int i = 0; i < people.size(); ++i) {
//            for (auto &j: people[i]) {
//                people_int[i].push_back(to_int[j]);
//                who_has_skill[to_int[j]].push_back(i);
//            }
//        }
//        vector<int> res;
//        int min_len = INT32_MAX;
//        auto f = [&](auto &&f, int skill_index, unordered_set<int> &persons) {
//            if (std::any_of(who_has_skill[skill_index].begin(), who_has_skill[skill_index].end(),
//                            [&](const auto &item) {
//                                return persons.count(item);
//                            })) {
//                if (skill_index + 1 >= nums_skills) {
//                    if (min_len > persons.size()) {
//                        min_len = persons.size();
//                        res = std::move(vector<int>(persons.begin(), persons.end()));
//                    }
//                    return;
//                } else f(f, skill_index + 1, persons);
//            } else {
//                if (skill_index + 1 >= nums_skills) {
//                    if (min_len > persons.size() + 1) {
//                        min_len = persons.size() + 1;
//                        res = std::move(vector<int>(persons.begin(), persons.end()));
//                        res.push_back(who_has_skill[skill_index][0]);
//                    }
//                    return;
//                } else
//                    for (auto i: who_has_skill[skill_index]) {
//                        persons.insert(i);
//                        f(f, skill_index + 1, persons);
//                        persons.erase(i);
//                    }
//            }
//        };
//        unordered_set<int> persons;
//        f(f, 0, persons);
//        return res;

        int n = req_skills.size();
        unordered_map<int, vector<int>> res;
        res.reserve(1 << n);
        res[0] = {};
        unordered_map<string, int> skill_map;
        for (int i = 0; i < req_skills.size(); i++)
            skill_map[req_skills[i]] = i;

        for (int i = 0; i < people.size(); i++) {
            int curr_skill = 0;
            for (int j = 0; j < people[i].size(); j++)
                curr_skill |= 1 << skill_map[people[i][j]];

            for (auto it = res.begin(); it != res.end(); it++) {
                int comb = it->first | curr_skill;
                if (res.find(comb) == res.end() || res[comb].size() > 1 + res[it->first].size()) {
                    res[comb] = it->second;
                    res[comb].push_back(i);
                }
            }
        }
        return res[(1 << n) - 1];
    }

    int largestPathValue(string colors, vector<vector<int>> &edges) {
        int n = colors.size();
        vector<int> in(n, 0);
        vector<bool> visited(n, false);
        vector<vector<int>> graph(n, vector<int>());
        for (auto &i: edges) {
            graph[i[0]].push_back(i[1]);
            ++in[i[1]];
        }
        vector<vector<int>> dp(n, vector<int>(26, 0));
        auto f = [&](int prev, int node) -> void {
            for (int i = 0; i <= 'z' - 'a'; ++i) {
                dp[node][i] = max(dp[node][i], dp[prev][i]);
            }
            dp[node][colors[node] - 'a'] = max(dp[node][colors[node] - 'a'], dp[prev][colors[node] - 'a'] + 1);
        };
        stack<int> s;
        for (int i = 0; i < n; ++i) {
            if (!in[i]) {
                s.push(i);
                dp[i][colors[i] - 'a'] = 1;
            }
        }
        while (!s.empty()) {
            auto top = s.top();
            s.pop();
            visited[top] = true;
            for (auto i: graph[top]) {
                --in[i];
                f(top, i);
                if (!in[i]) s.push(i);
            }
        }
        if (std::any_of(visited.begin(), visited.end(), [&](const auto &item) {
            return !item;
        }))
            return -1;
        int res = 0;
        for (auto &i: dp) {
            for (auto j: i) {
                res = max(res, j);
            }
        }
        return res;
    }

    bool isValid(string s) {
        stack<char> st;
        auto f = [](char a, char b) -> bool {
            return (a == '[' && b == ']') || (a == '{' && b == '}') || (a == '(' && b == ')');
        };
        for (auto i: s) {
            if (i == '{' || i == '[' || i == '(') {
                st.push(i);
            } else {
                if (st.empty() || !f(st.top(), i)) {
                    return false;
                }
                st.pop();
            }
        }
        return st.empty();
    }

    vector<int> nextLargerNodes(ListNode *head) {
        vector<pair<int, int>> st;
        vector<int> res;
        auto pt = head;
        int index = 0;
        while (pt) {
            int val = pt->val;
            res.push_back(0);
            while (!st.empty() && val > st.back().second) {
                res[st.back().first] = val;
                st.pop_back();
            }
            st.push_back(pair<int, int>(index, val));
            ++index;
            pt = pt->next;
        }
        return res;
    }

    string removeStars(const string &s) {
        string res;
        for (auto i: s) {
            if (i == '*' && !res.empty()) res.pop_back();
            else res.push_back(i);
        }
        return std::move(res);
    }

    bool isRobotBounded(const string &instructions) {
        int x = 0, y = 0, dir = 0;
        int xy[5] = {0, 1, 0, -1, 0};
        for (int i = 0; i < instructions.size(); ++i) {
            if (instructions[i] == 'G') {
                x += xy[dir];
                y += xy[dir + 1];
            } else {
                dir += (instructions[i] == 'L') ? 3 : 1;
                dir %= 4;
            }
        }
        if (dir == 0) {
            return x == 0 && y == 0;
        }
        return true;
    }

    string simplifyPath(const string &path) {
        vector<string> res;
        for (int i = 0; i < path.size();) {
            int j = i + 1;
            while (j < path.size() && path[j] != '/') {
                ++j;
            }
            if (j == i + 1 || (j == i + 2 && path[j - 1] == '.')) {
            } else if (j == i + 3 && path[j - 1] == '.' && path[j - 2] == '.') {
                if (!res.empty()) {
                    res.pop_back();
                }
            } else {
                res.push_back(path.substr(i + 1, j - i - 1));
            }
            i = j;
        }
        if (res.empty()) {
            return "/";
        }
        string ans;
        for (auto &i: res) {
            ans.push_back('/');
            ans.append(i);
        }
        return std::move(ans);
    }

    int longestDecomposition(const string &text) {
        int res = 0;
        auto i = text.begin();
        auto j = text.end() - 1;
        deque<char> a;
        deque<char> b;
        auto cmp = [&]() -> bool {
            for (auto i = 0; i < a.size(); ++i) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        };
        while (i < j) {
            a.push_back(*i++);
            b.push_front(*j--);
            if (cmp()) {
                res += 2;
                a.clear();
                b.clear();
            }
        }
        return res + (!a.empty() || i == j);
    }

    bool validateStackSequences(vector<int> &pushed, vector<int> &popped) {
        stack<int> s;
        int index = 0;
        for (auto i: pushed) {
            if (i != popped[index]) {
                s.push(i);
            } else {
                ++index;
                while (!s.empty() && index < pushed.size() && popped[index] == s.top()) {
                    s.pop();
                    ++index;
                }
            }
        }
        while (!s.empty()) {
            if (index < pushed.size() && s.top() == popped[index]) {
                s.pop();
                ++index;
            } else {
                return false;
            }
        }
        return true;
    }

    int longestPalindromeSubseq(const string &s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (auto i = 0; i < n; ++i) {
            dp[i][i] = 1;
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dp[i][j] = max(dp[i][j - 1], dp[i + 1][j]);
                if (s[i] == s[j]) {
                    dp[i][j] = max(dp[i][j], 2 + dp[i + 1][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    vector<bool> camelMatch(vector<string> &queries, const string &pattern) {
        vector<bool> res(queries.size(), false);
        for (int i = 0; i < queries.size(); ++i) {
            int index = 0;
            for (auto &c: queries[i]) {
                if (index >= pattern.size()) {
                    if (c >= 'A' && c <= 'Z') {
                        res[i] = false;
                    }
                } else if (pattern[index] == c) {
                    ++index;
                    if (index == pattern.size()) {
                        res[i] = true;
                    }
                } else if (c <= 'Z' && c >= 'A') {
                    res[i] = false;
                    break;
                }
            }
        }
        return std::move(res);
    }

    int minDays(vector<vector<int>> &grid) {
        const int xy[5] = {0, 1, 0, -1, 0};
        int n = grid.size();
        int m = grid.begin()->size();
        vector<vector<bool>> visited(n, vector<bool>(m, false));
        auto cal_connected_component = [&](auto &&cal_connected_component, int x, int y) -> int {
            visited[x][y] = true;
            for (int k = 0; k < 4; ++k) {
                int xx = x + xy[k];
                int yy = y + xy[k + 1];
                if (xx >= 0 && yy >= 0 && xx < n && yy < m) {
                    if (!visited[xx][yy] && grid[xx][yy]) {
                        cal_connected_component(cal_connected_component, xx, yy);
                    }
                }
            }
            return 1;
        };
        int con_com = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] && !visited[i][j]) {
                    con_com += cal_connected_component(cal_connected_component, i, j);
                }
            }
        }
        if (con_com ^ 1) {
            return 0;
        }
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < m; ++b) {
                if (grid[a][b]) {
                    grid[a][b] = 0;
                    con_com = 0;
                    std::fill(visited.begin(), visited.end(), vector<bool>(m, false));
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < m; ++j) {
                            if (grid[i][j] && !visited[i][j]) {
                                con_com += cal_connected_component(cal_connected_component, i, j);
                            }
                        }
                    }
                    if (con_com ^ 1) {
                        return 1;
                    }
                    grid[a][b] = 1;
                }
            }
        }
        return 2;
    }

    int maxValueOfCoins(vector<vector<int>> &piles, int k) {
        for (auto &i: piles) {
            for (int j = 1; j < i.size(); ++j) {
                i[j] += i[j - 1];
            }
        }
        int n = piles.size();
        vector<int> dp(k + 1, 0);
        int j = 1;
        for (; j <= k && j <= piles[0].size(); ++j) {
            dp[j] = piles[0][j - 1];
        }
        for (; j <= k; ++j) {
            dp[j] = dp[j - 1];
        }
        for (int i = 1; i < n; ++i) {
            vector<int> new_dp(k + 1, 0);
            for (int j = 1; j <= k; ++j) {
                new_dp[j] = max(dp[j], new_dp[j - 1]);
                for (int m = 1; m <= j && m <= piles[i].size(); ++m) {
                    new_dp[j] = max(new_dp[j],
                                    dp[j - m] + piles[i][m - 1]);
                }
            }
            dp.swap(new_dp);
        }
        return dp[k];
    }

    int numWays(vector<string> &words, const string &target) {
        int n = target.size();
        int m = words.begin()->size();
        int mod = 1000000007;
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        vector<vector<int>> count(m, vector<int>(26, 0));
        for (const string &word: words) {
            for (int i = 0; i < m; i++) {
                count[i][word[i] - 'a']++;
            }
        }
        for (int i = 0; i < m; i++) {
            // 1D DP array optimization, dp[j+1] depends on previous dp[j].
            // So we must update dp[j+1] before dp[j]. As a result, j begins at n-1,
            // and decreases to 0, and can ensure when update dp[j+1], dp[j] is previous result.
            for (int j = n - 1; j >= 0; j--) {
                dp[j + 1] += (int) ((long) dp[j] * count[i][target[j] - 'a'] % mod);
                dp[j + 1] %= mod;
            }
        }
        return dp[n];
    }

    vector<bool> kidsWithCandies(vector<int> &candies, int extraCandies) {
        int max_num = candies[0];
        for (int i = 1; i < candies.size(); ++i) {
            if (max_num < candies[i]) max_num = candies[i];
        }
        vector<bool> res(candies.size(), true);
        for (int i = 0; i < candies.size(); ++i) {
            if (extraCandies + candies[i] < max_num) res[i] = false;
        }
        return std::move(res);
    }

    string mergeAlternately(const string &word1, const string &word2) {
        string res;
        int a = 0;
        int b = 0;
        while (a < word1.size() || b < word2.size()) {
            if (a < word1.size()) {
                res.push_back(word1[a++]);
            }
            if (b < word2.size()) {
                res.push_back(word2[b++]);
            }
        }
        return std::move(res);
    }

    int maxAncestorDiff(TreeNode *root) {
        auto dfs = [&](auto &&dfs, TreeNode *cur, int mi, int ma) -> int {
            if (cur) {
                mi = min(cur->val, mi);
                ma = max(cur->val, ma);
                return max(dfs(dfs, cur->left, mi, ma), dfs(dfs, cur->right, mi, ma));
            } else return ma - mi;
        };
        return dfs(dfs, root, INT32_MAX, INT32_MIN);
    }

    int longestZigZag(TreeNode *root) {
        int res = 0;
        auto dfs = [&](auto &&dfs, int left, int right, TreeNode *cur) -> void {
            if (cur->left) {
                dfs(dfs, 0, left + 1, cur->left);
            }
            if (cur->right) {
                dfs(dfs, right + 1, 0, cur->right);
            }
            res = max(res, max(left, right));
        };
        dfs(dfs, 0, 0, root);
        return res;
    }

    int widthOfBinaryTree(TreeNode *root) {
        int res = 1;
        using item = pair<TreeNode *, unsigned int>;
        vector<item> q;
        vector<item> temp;
        q.emplace_back(root, 1);
        while (!q.empty()) {
            temp.clear();
            int base = q.front().second;
            int max_num = q.front().second;
            std::for_each(q.begin(), q.end(), [&](const auto &item) {
                auto [front, num] = item;
                num = num - base;
                max_num = num;
                if (front->left) {
                    temp.emplace_back(front->left, 2 * num - 1);
                }
                if (front->right) {
                    temp.emplace_back(front->right, 2 * num);
                }
            });
            res = max(res, max_num + 1);
            q.swap(temp);
        }
        return res;
    }

    int lengthOfLIS(vector<int> &nums) {
        int n = nums.size();
        vector<int> tails;
        int size = 0;
        for (auto num: nums) {
            if (tails.empty() || tails.back() < num) {
                tails.push_back(num);
            } else {
                int i = 0, j = tails.size();
                while (i != j) {
                    int m = (i + j) >> 1;
                    if (tails[m] < num)
                        i = m + 1;
                    else
                        j = m;
                }
                tails[i] = num;
            }
//            This replacement technique works because replaced elements don't matter to us
//            We only used end elements of existing lists to check if they can be extended otherwise form newer lists
//            And since we have replaced a bigger element with smaller one it won't affect the
//            step of creating new list after taking some part of existing list
        }
        return tails.size();
    }

    int makeArrayIncreasing(vector<int> &arr1, vector<int> &arr2) {
        std::sort(arr2.begin(), arr2.end());
        vector<vector<int>> dp(arr1.size() + 1, vector<int>(arr2.size() + 1, -1));
        auto solve = [&](auto &&solve, int i, int j, int prev) -> int {
            if (i >= arr1.size()) {
                return 0;
            }
            j = upper_bound(arr2.begin() + j, arr2.end(), prev) - arr2.begin();
            if (arr2.size() <= j && arr1[i] < prev) {
                return 3000;
            }
            if (dp[i][j] != -1) {
                return dp[i][j];
            }
            int take_swap = 3000, notake = 3000;
            if (j < arr2.size()) {
                take_swap = 1 + solve(solve, i + 1, j + 1, arr2[j]);
            }
            if (arr1[i] > prev) {
                notake = solve(solve, i + 1, j, arr1[i]);
            }
            return dp[i][j] = min(take_swap, notake);
        };
        int res = solve(solve, 0, 0, -1);
        return res <= arr1.size() ? res : -1;
    }

//    string longestPalindrome(const string &s) {
//        int len = s.size();
//        if (len == 1) return s;
//        int start = 0;
//        int res_len = 1;
//        vector<vector<bool>> dp(2, vector<bool>(len, true));
//        for (int i = 0; i < len - 1; ++i) {
//            if (s[i] == s[i + 1]) {
//                dp[0][i] = true;
//                start = i;
//                res_len = 2;
//            } else {
//                dp[0][i] = false;
//            }
//        }
//        bool flag = true;
//        auto proc = [&](int i) {
//            for (int k = 0; k <= len - i; ++k) {
//                if (dp[i % 2][k + 1] && s[k] == s[k + i - 1]) {
//                    dp[i % 2][k] = true;
//                    flag = true;
//                    res_len = i;
//                    start = k;
//                } else {
//                    dp[i % 2][k] = false;
//                }
//            }
//        };
//        for (int i = 3; i <= len && flag; ++i) {
//            flag = false;
//            proc(i);
//            ++i;
//            proc(i);
//        }
//        return s.substr(start, res_len);
//    }
    string longestPalindrome(const string &s) {
        int n = s.size();
        int res_r = 0;
        int res_l = 0;
        for (int i = 0; i < n; ++i) {
            int j = i;
            while (j < n - 1 && s[j + 1] == s[i]) {
                ++j;
            }
            int l = i, r = j;
            while (l > 0 && r < n - 1 && s[l - 1] == s[r + 1]) {
                --l;
                ++r;
            }
            if (res_r - res_l < r - l) {
                res_r = r;
                res_l = l;
            }
            i = j;
        }
        return std::move(s.substr(res_l, res_r - res_l + 1));
    }

    int profitableSchemes(int n, int minProfit, vector<int> &group, vector<int> &profit) {
        const int mod = 1e9 + 7;
        vector<vector<int>> dp(minProfit + 1, vector<int>(n + 1));
        dp[0][0] = 1;
        for (int k = 0; k < group.size(); k++) {
            int g = group[k], p = profit[k];
            for (int i = minProfit; i >= 0; i--) {
                for (int j = n - g; j >= 0; --j) {
                    int newProfit = min(minProfit, i + p);
                    dp[newProfit][j + g] = (dp[newProfit][j + g] + dp[i][j]) % mod;
                }
            }
        }
        int sum = 0;
        for (int i = 0; i <= n; i++) {
            sum += dp[minProfit][i];
            sum %= mod;
        }
        return sum;
    }

//    let say array be [a,b,c,d]
//    answer = (a+b)-(c+d) OR
//    answer = a-(b+c+d) Or
//    answer = (d+b)-(a+c) and so on.. any combination could be possible
//    notice that in general I can say that
//    answer = S1-S2
//    where S1 is sum of some of the numbers and S2 is sum of rest of numbers
//    also note that S1+S2 = SUM (sum of all numbers)
//    S1 >= S2 beacuse negative answer is not possible
//    now we have to minimise answer
//    answer = SUM - 2*S2 (Just substituting S1 by SUM-S2)
//    To minimise answer S2 has to be maximum
//    Now, max value of S2 is SUM/2 (bigger than this and answer would be negative which is not possible)
//    so the question reduces to find closest sum (sum of numbers) to (SUM/2)
//    now this could be understood as subset sum problem or 0/1 knapsack problem

    int lastStoneWeightII(vector<int> &stones) {
        int n = stones.size();
        int total = 0;
        for (int i = 0; i < n; i++) {
            total += stones[i];
        }
        int sum = total >> 1;
        vector<vector<int>> dp(n + 1, vector<int>(sum + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= sum; j++) {
                if (stones[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - stones[i - 1]] + stones[i - 1]);
                }
            }
        }
        return total - (2 * dp[n][sum]);
    }

    int minAbsDifference(vector<int> &nums, int goal) {
        int n = nums.size();
        vector<int> first;
        vector<int> second;
        auto generate = [&](auto &&generate, int i, int end, int sum, vector<int> &listOfSubsetSums) -> void {
            if (i == end) {
                listOfSubsetSums.push_back(sum); //add
                return;
            }
            generate(generate, i + 1, end, sum + nums[i], listOfSubsetSums);
            generate(generate, i + 1, end, sum, listOfSubsetSums);
        };
        generate(generate, 0, n / 2, 0, first); //generate all possible subset sums from half the array
        generate(generate, n / 2, n, 0, second);//generate all possible subset sums from the second half of the array
        std::sort(first.begin(), first.end());
        int ans = INT32_MAX;
        auto binarySearch = [&](vector<int> &first, int target) {
            int i = 0;
            int j = first.size() - 1;
            int mid = (i + j) >> 1;
            while (i <= j) {
                mid = (i + j) >> 1;
                if (first[mid] == target) {
                    return mid;
                } else if (first[mid] < target) {
                    i = mid + 1;
                } else {
                    j = mid - 1;
                }
            }
            return i;
        };
        for (auto secondSetSum: second) {
            int left = goal - secondSetSum; // How far off are we from the desired goal?
            if (first[0] > left) { // all subset sums from first half are too big => Choose the smallest
                ans = min(ans, abs((first[0] + secondSetSum) - goal));
                continue;
            }
            if (first[first.size() - 1] <
                left) { // all subset sums from first half are too small => Choose the largest
                ans = min(ans, abs((first[(first.size() - 1)] + secondSetSum) - goal));
                continue;
            }
            int pos = binarySearch(first, left);
            if (first[pos] == left)
                return 0;
            ans = min(ans, abs(
                    secondSetSum + first[pos - 1] - goal)); // Checking for the floor value (largest sum < goal)
            ans = min(ans, abs(
                    secondSetSum + first[pos] - goal)); //Checking for the ceiling value (smallest sum > goal)
        }
        return ans;
    }

    int minimumDifference(vector<int> &nums) {
        auto binarySearch = [&](vector<int> &arr, int target) {
            int i = 0;
            int j = arr.size() - 1;
            int mid = (i + j) >> 1;
            while (i <= j) {
                mid = (i + j) >> 1;
                if (arr[mid] == target) {
                    return mid;
                } else if (arr[mid] < target) {
                    i = mid + 1;
                } else {
                    j = mid - 1;
                }
            }
            return i;
        };
        int sum = 0;
        int n = nums.size();
        if (n == 2) {
            return ::abs(nums[1] - nums[0]);
        }
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
        }
        int len = n / 2;
        int target = sum >> 1;
        vector<vector<int>> first(len + 1, vector<int>());
        vector<vector<int>> second(len + 1, vector<int>());
        first[0].push_back(0);
        second[0].push_back(0);
        auto generate = [&](auto &&generate, vector<vector<int>> &cur_set, int index, int end, int size,
                            int sum) -> void {
            if (size <= len) {
                if (size > 0) cur_set[size].push_back(sum);
            }
            if (size == len || index >= end) {
                return;
            }
            generate(generate, cur_set, index + 1, end, size + 1, sum + nums[index]);
            generate(generate, cur_set, index + 1, end, size, sum);
        };
        generate(generate, first, 0, len, 0, 0);
        generate(generate, second, len, n, 0, 0);
        int res = INT32_MAX >> 1;
        for (int i = 0; i < len; ++i) {
            std::sort(first[i].begin(), first[i].end());
        }
        for (int i = 0; i <= len; ++i) {
            for (auto k: second[i]) {
                int left = target - k;
                auto &left_vector = first[len - i];
                if (left_vector[0] >= left) {
                    res = min(res, ::abs(2 * (k + left_vector[0]) - sum));
                    continue;
                }
                if (left_vector.back() <= left) {
                    res = min(res, ::abs(sum - 2 * (k + left_vector.back())));
                    continue;
                }
                int pos = binarySearch(left_vector, left);
                int search_res = left_vector[pos];
                if (search_res == left) {
                    return sum - 2 * target;
                } else {
                    res = min(res, ::abs(2 * (k + search_res) - sum));
                    res = min(res, ::abs(sum - 2 * (k + left_vector[pos - 1])));
                }
            }
        }
        return res;
    }

    int minInsertions(const string &s) {
        int n = s.size();
        vector<vector<int>> dp(2, vector<int>(n, 0));
        for (int i = 2; i <= n; ++i) {
            for (int j = 0; j <= n - i; ++j) {
                if (s[j] == s[j + i - 1]) {
                    dp[i % 2][j] = dp[i % 2][j + 1];
                } else {
                    dp[i % 2][j] = 1 + min(dp[(i - 1) % 2][j], dp[(i - 1) % 2][j + 1]);
                }
            }
        }
        return dp[n % 2][0];
    }

    int longestArithSeqLength(vector<int> &nums) {
        int n = nums.size();
        int res = 1;
        vector<vector<int>> dp(n + 1, vector<int>(1001, 1));
        int maxLen = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int diff = nums[j] - nums[i];
                dp[j][diff + 500] = 1 + dp[i][diff + 500];
                res = max(res, dp[j][diff + 500]);
            }
        }
        return res;
    }

    vector<int> getOrder(vector<vector<int>> &tasks) {
        vector<int> res;
        int n = tasks.size();
        for (int i = 0; i < n; ++i) {
            tasks[i].push_back(i);
        }
        std::sort(tasks.begin(), tasks.end(), [](const vector<int> &a, const vector<int> &b) {
            return a[0] < b[0];
        });
        struct task {
            int index;
            int consumed;

            task(int a, int b) : index(a), consumed(b) {}

            bool operator<(const task &a) const {
                return consumed > a.consumed || (consumed == a.consumed && index > a.index);
            }
        };
        priority_queue<task> pq;
        long long int time = 0;
        for (int i = 0; i < n;) {
            if (time >= tasks[i][0]) {
                pq.push({tasks[i][2], tasks[i][1]});
                ++i;
            } else {
                if (pq.empty()) {
                    time = tasks[i][0];
                } else {
                    auto [index, consumed_time] = pq.top();
                    res.push_back(index);
                    time += consumed_time;
                    pq.pop();
                }
            }
        }
        while (!pq.empty()) {
            auto [index, consumed_time] = pq.top();
            res.push_back(index);
            time += consumed_time;
            pq.pop();
        }
        return std::move(res);
    }

    int numSubarraysWithSum(vector<int> &nums, int goal) {
        int res = 0;
        int n = nums.size();
        for (int i = 1; i < n; ++i) {
            nums[i] += nums[i - 1];
        }
        vector<int> times(nums[n - 1] + 1, 0);
        ++times[0];
        for (int i = 0; i < n; ++i) {
            if (nums[i] >= goal) {
                res += times[nums[i] - goal];
            }
            ++times[nums[i]];
        }
        return res;
    }

    vector<int> shortestAlternatingPaths(int n, vector<vector<int>> &redEdges, vector<vector<int>> &blueEdges) {
        vector<vector<int>> redGraph(n, vector<int>());
        vector<vector<int>> blueGraph(n, vector<int>());
        for (auto &i: redEdges) {
            redGraph[i[0]].push_back(i[1]);
        }
        for (auto &i: blueEdges) {
            blueGraph[i[0]].push_back(i[1]);
        }
        vector<int> res(n, INT32_MAX >> 1);
        res[0] = 0;
        auto proc = [&](bool flag) {
            vector<int> next{0};
            vector<vector<bool>> visited(2, vector<bool>(n, false));
            visited[flag][0] = true;
            int len = 0;
            while (!next.empty()) {
                vector<int> temp;
                ++len;
                for (auto v: next) {
                    if (flag) {
                        for (auto j: redGraph[v]) {
                            if (!visited[flag][j]) {
                                res[j] = min(res[j], len);
                                temp.push_back(j);
                                visited[flag][j] = true;
                            }
                        }
                    } else {
                        for (auto j: blueGraph[v]) {
                            if (!visited[flag][j]) {
                                res[j] = min(res[j], len);
                                temp.push_back(j);
                                visited[flag][j] = true;
                            }
                        }
                    }
                }
                temp.swap(next);
                flag = !flag;
            }
        };
        proc(true);
        proc(false);
        for (int i = 0; i < n; ++i) {
            if (res[i] == INT32_MAX >> 1) {
                res[i] = -1;
            }
        }
        return std::move(res);
    }

    int kthLargestValue(vector<vector<int>> &matrix, int k) {
        priority_queue<int> pq;
        int m = matrix.size();
        int n = matrix.begin()->size();
        auto push_into_pq = [&](int value) {
            pq.push(-value);
            while (pq.size() > k) {
                pq.pop();
            }
        };
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i) matrix[i][j] ^= matrix[i - 1][j];
                if (j) matrix[i][j] ^= matrix[i][j - 1];
                if (i && j) matrix[i][j] ^= matrix[i - 1][j - 1];
                push_into_pq(matrix[i][j]);
            }
        }
        return -pq.top();
    }

    std::string longestDupSubstring(std::string s) {
        auto RabinKarp = [](const std::string &text, int M, long long q) -> std::pair<bool, std::string> {
            if (M == 0) return {true, ""};
            long long h = 1, t = 0, d = 256;
            std::unordered_map<long long, std::vector<int>> dic;
            for (int i = 0; i < M - 1; ++i) { h = (h * d) % q; }
            for (int i = 0; i < M; ++i) { t = (d * t + text[i]) % q; }
            dic[t].push_back(0);
            for (int i = 0; i < static_cast<int>(text.size()) - M; ++i) {
                t = (d * (t - text[i] * h) + text[i + M]) % q;
                if (t < 0) t += q;
                for (int j: dic[t]) {
                    if (text.substr(i + 1, M) == text.substr(j, M)) {
                        return {true, text.substr(j, M)};
                    }
                }
                dic[t].push_back(i + 1);
            }
            return {false, ""};
        };
        int beg = 0, end = s.size();
        long long q = 1e9 + 7;
        std::string found = "";
        while (beg + 1 < end) {
            int mid = beg + (end - beg) / 2;
            auto [isFound, candidate] = RabinKarp(s, mid, q);
            if (isFound) {
                beg = mid;
                found = candidate;
            } else { end = mid; }
        }
        return found;
    }

    int numberOfArrays(const string &s, int k) {
        int n = s.size();
        const int mod = 1e9 + 7;
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 0; i < n; ++i) {
            if (s[i] == '0') {
                continue;
            }
            long long int num = s[i] - '0';
            if (num <= k) {
                dp[i + 1] = (dp[i + 1] + dp[i]) % mod;
            }
            for (int j = i + 1; j < n; ++j) {
                num = num * 10 + s[j] - '0';
                if (num <= k)
                    dp[j + 1] = (dp[j + 1] + dp[i]) % mod;
                else break;
            }
        }
        return dp[n];
    }

    long long minCost(vector<int> &basket1, vector<int> &basket2) {
        unordered_map<int, int> map;
        for (auto i: basket1) {
            ++map[i];
        }
        for (auto i: basket2) {
            --map[i];
            if (map[i] == 0) {
                map.erase(i);
            }
        };
        int min_num = min(*std::min_element(basket1.begin(), basket1.end()),
                          *std::min_element(basket2.begin(), basket2.end()));
        long long int res = 0;
        vector<pair<int, int>> more;
        vector<pair<int, int>> less;
        int more_size = 0;
        int less_size = 0;
        for (auto &i: map) {
            if (i.second & 1) {
                return -1;
            }
            if (i.second > 0) {
                more_size += i.second;
                more.emplace_back(i);
            } else {
                less_size -= i.second;
                less.emplace_back(i);
            }
        }
        if (more_size != less_size) {
            return -1;
        }
        std::sort(more.begin(), more.end(), [](const pair<int, int> &a, const pair<int, int> &b) -> bool {
            return a.first < b.first;
        });
        std::sort(less.begin(), less.end(), [](const pair<int, int> &a, const pair<int, int> &b) -> bool {
            return a.first < b.first;
        });
        int i = 0;
        int j = 0;
        for (int k = 0; k < more_size / 2; ++k) {
            if (more[i].first < less[j].first) {
                res += min(more[i].first, 2 * min_num);
                more[i].second -= 2;
                if (more[i].second == 0) {
                    ++i;
                }
            } else {
                res += min(2 * min_num, less[j].first);
                less[j].second += 2;
                if (less[j].second == 0) {
                    ++j;
                }
            }
        }
        return res;
    }

    int lastStoneWeight(vector<int> &stones) {
        int n = stones.size();
        int last = n - 1;
        auto up = [&](int p) {
            while (p > 1 && stones[p - 1] > stones[p / 2 - 1]) {
                swap(stones[p - 1], stones[p / 2 - 1]);
                p /= 2;
            }
        };
        auto down = [&](int p) {
            while (p * 2 - 1 <= last) {
                int t = p * 2;
                if (t <= last && stones[t] > stones[t - 1]) t++;
                if (stones[t - 1] <= stones[p - 1]) break;
                std::swap(stones[p - 1], stones[t - 1]);
                p = t;
            }
        };
        auto push = [&](int val) {
            ++last;
            stones[last] = val;
            int i = last;
            up(last + 1);
        };
        auto pop = [&]() -> int {
            if (last >= 0) {
                int res = stones[0];
                swap(stones[0], stones[last]);
                --last;
                down(1);
                return res;
            } else {
                return -1;
            }
        };
        for (int i = n; i >= 1; i--) down(i);
        while (last > 0) {
            auto a = pop();
            auto b = pop();
            if (a ^ b) {
                push(::abs(a - b));
            }
        }
        return last == 0 ? stones[0] : 0;
    }

    int minMutation(const string &startGene, const string &endGene, vector<string> &bank) {
        auto intoInt = [](char a) -> int {
            switch (a) {
                case 'A':
                    return 0;
                case 'C':
                    return 1;
                case 'G':
                    return 2;
                case 'T':
                    return 3;
                default:
                    return -1;
            }
        };
        auto hash = [&](const string &s) -> int {
            int res = 0;
            for (int i = 0; i < s.size(); ++i) {
                res += intoInt(s[i]) * (1 << (2 * i));
            }
            return res;
        };
        unordered_set<int> bank_int;
        for (auto &i: bank) {
            bank_int.insert(hash(i));
        }
        queue<int> q;
        int res = 0;
        q.push(hash(startGene));
        int end = hash(endGene);
        while (!q.empty()) {
            queue<int> temp;
            while (!q.empty()) {
                int top = q.front();
                q.pop();
                if (top == end) {
                    return res;
                }
                for (int i = 0; i < startGene.size(); ++i) {
                    for (int c = 0; c < 4; ++c) {
                        int new_hash =
                                top - (top >> (2 * i)) % 4 * (1 << (2 * i)) + (c) * (1 << (2 * i));
                        if (new_hash == top) {
                            continue;
                        } else if (bank_int.count(new_hash)) {
                            if (new_hash == end) {
                                return res + 1;
                            }
                            temp.push(new_hash);
                            bank_int.erase(new_hash);
                        }
                    }
                }
            }
            temp.swap(q);
            ++res;
        }
        return -1;
    }

    bool increasingTriplet(vector<int> &nums) {
        int min1 = INT32_MAX;
        int min2 = INT32_MAX;
        for (auto i: nums) {
            if (i <= min1) {
                min1 = i;
            } else if (i <= min2) {
                min2 = i;
            } else {
                return true;
            }
        }
        return false;
    }

    int addDigits(int num) {
        int sum = 0;
        while (num > 9) {
            for (auto i: to_string(num)) {
                sum += (i - '0');
            }
            num = sum;
            sum = 0;
        }
        return num;
    }

    double (*bulbSwitch)(double) = &sqrt;

    long long maximumSubsequenceCount(const string &text, const string &pattern) {
        long long sum = 0;
        long long res = 0;
        int num0 = 0;
        int num1 = 0;
        for (int i = 0; i < text.size(); ++i) {
            if (text[i] == pattern[0]) {
                ++sum;
                ++num0;
            } else if (text[i] == pattern[1]) {
                res += sum;
                ++num1;
            }
        }
        if (pattern[0] == pattern[1]) {
            res = (sum - 1) * sum / 2;
        }
        if (num1 > num0) {
            res += num1;
        } else {
            res += num0;
        }
        return res;
    }

    int numSimilarGroups(vector<string> &strs) {
        int n = strs.size();
        int m = strs.begin()->size();
        vector<vector<int>> graph(n, vector<int>());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int diff = 0;
                int k;
                for (k = 0; k < m; ++k) {
                    if (strs[i][k] != strs[j][k]) {
                        ++diff;
                        if (diff >= 3) {
                            break;
                        }
                    }
                }
                if (k == m) {
                    graph[i].push_back(j);
                    graph[j].push_back(i);
                }
            }
        }
        vector<int> parent(n);
        vector<int> size(n);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            size[i] = 1;
        }
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
            for (auto j: graph[i]) {
                uni(i, j);
            }
        }
        unordered_set<int> sets;
        for (int i = 0; i < n; ++i) {
            sets.insert(find(parent[i]));
        }
        return sets.size();
    }

    long long minimumMoney(vector<vector<int>> &transactions) {
        for (auto &i: transactions) {
            i[1] -= i[0];
        }
        long long int max_cost_pos = 0;
        long long int neg_sum = 0;
        vector<int> neg_index;
        for (int i = 0; i < transactions.size(); ++i) {
            if (transactions[i][1] >= 0) {
                max_cost_pos = max(max_cost_pos, (long long int) transactions[i][0]);
            } else {
                neg_sum += transactions[i][1];
                neg_index.push_back(i);
            }
        }
        long long int res = max_cost_pos - neg_sum;
        for (auto i: neg_index) {
            res = max(res, transactions[i][0] - neg_sum + transactions[i][1]);
        }
        return res;
    }

    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>> &edgeList, vector<vector<int>> &queries) {
        vector<int> parent(n);
        vector<int> size(n);
        for (int i = 0; i < queries.size(); ++i) {
            queries[i].push_back(i);
        }
        std::sort(edgeList.begin(), edgeList.end(), [](const vector<int> &a, const vector<int> &b) -> bool {
            return a[2] < b[2];
        });
        std::sort(queries.begin(), queries.end(), [](const vector<int> &a, const vector<int> &b) -> bool {
            return a[2] < b[2];
        });
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
        vector<bool> res(queries.size());
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            size[i] = 1;
        }
        int index = 0;
        for (auto &i: queries) {
            while (index < edgeList.size() && edgeList[index][2] < i[2]) {
                uni(edgeList[index][0], edgeList[index][1]);
                ++index;
            }
            res[i[3]] = ((find(i[0]) == find(i[1])));
        }
        return std::move(res);
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

    int minimumMountainRemovals(vector<int> &nums) {
        int origin_len = nums.size();
        vector<int> dp(nums.size());
        vector<int> queue;
        for (int i = 0; i < nums.size(); ++i) {
            auto index = std::lower_bound(queue.begin(), queue.end(), nums[i]);
            dp[i] = index - queue.begin();
            if (index == queue.end()) {
                queue.push_back(nums[i]);
            } else {
                *index = nums[i];
            }
        }
        vector<int> rdp(nums.size());
        queue.clear();
        for (int i = nums.size() - 1; i >= 0; --i) {
            auto index = std::lower_bound(queue.begin(), queue.end(), nums[i]);
            rdp[i] = index - queue.begin();
            if (index == queue.end()) {
                queue.push_back(nums[i]);
            } else {
                *index = nums[i];
            }
        }
        int sum = 0;
        for (int i = 0; i < dp.size(); ++i) {
            if (dp[i] > 0 && rdp[i] > 0)
                sum = max(sum, dp[i] + rdp[i] + 1);
        }
        return origin_len - sum;
    }

    double average(vector<int> &salary) {
        int n = salary.size();
        int min = ::min(salary[0], salary[1]);
        int max = ::max(salary[0], salary[1]);
        double sum = 0.0;
        int i;
        for (int k = 2; k < n; ++k) {
            i = salary[k];
            if (i > max) {
                sum += max;
                max = i;
            } else if (i < min) {
                sum += min;
                min = i;
            } else
                sum += i;
        }
        return sum / (n - 2);
    }

    int maxEnvelopes(vector<vector<int>> &envelopes) {
        std::sort(envelopes.begin(), envelopes.end(), [](const vector<int> &a, const vector<int> &b) -> bool {
            return a[1] < b[1] || (a[1] == b[1] && a[0] > b[0]);
        });
        vector<int> q;
        for (int i = 0; i < envelopes.size(); ++i) {
            auto iter = std::lower_bound(q.begin(), q.end(), i, [&](int a, int b) -> bool {
                return envelopes[a][0] < envelopes[b][0];
            });
            if (iter == q.end()) {
                q.push_back(i);
            } else {
                *iter = i;
            }
        }
        return q.size();
    }

    vector<int> maxSumOfThreeSubarrays(vector<int> &nums, int k) {
        int n = nums.size();
        vector<int> sums(n - k + 1);
        sums[0] = std::accumulate(nums.begin(), nums.begin() + k, 0);
        for (int i = 1; i < sums.size(); ++i) {
            sums[i] = sums[i - 1] + nums[i - 1 + k] - nums[i - 1];
        }
        vector<int> right(n - k + 1);
        vector<int> left(n - k + 1);
        int max_sum = sums[n - k];
        right[n - k] = n - k;
        for (int i = n - k - 1; i >= 0; --i) {
            if (max_sum > sums[i]) {
                right[i] = right[i + 1];
            } else {
                right[i] = i;
                max_sum = sums[i];
            }
        }
        max_sum = sums[0];
        left[0] = 0;
        for (int i = 1; i <= n - k; ++i) {
            if (max_sum >= sums[i]) {
                left[i] = left[i - 1];
            } else {
                left[i] = i;
                max_sum = sums[i];
            }
        }
        vector<int> res = {0, k, k + k};
        max_sum = sums[0] + sums[k] + sums[k + k];
        for (int i = k; i <= n - k - k; ++i) {
            int sum = sums[i] + sums[left[i - k]] + sums[right[i + k]];
            if (sum > max_sum) {
                max_sum = sum;
                res = {left[i - k], i, right[i + k]};
            }
        }
        return res;
    }

    int arraySign(vector<int> &nums) {
        int sign = 1;
        for (auto i: nums) {
            if (i == 0) {
                return 0;
            } else if (i < 0) {
                sign = -sign;
            }
        }
        return sign;
    }

    int maxProfitII(vector<int> &prices) {
        int n = prices.size();
        vector<int> now(2), prev(2);
        prev[1] = -prices[0];
        prev[0] = 0;
        for (int i = 1; i < n; ++i) {
            now[1] = max(prev[0] - prices[i], prev[1]);
            now[0] = max(prev[1] + prices[i], prev[0]);
            now.swap(prev);
        }
        return prev[0];
    }

    int maxProfitIII(vector<int> &prices) {
        int n = prices.size();
        vector<int> left(n, 0);
        vector<int> right(n, 0);
        int min_num = prices[0];
        for (int i = 1; i < n; ++i) {
            min_num = min(prices[i], min_num);
            left[i] = max(left[i - 1], prices[i] - min_num);
        }
        int max_num = prices[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            max_num = max(prices[i], max_num);
            right[i] = max(right[i + 1], max_num - prices[i]);
        }
        int res = right[0];
        for (int i = 1; i < n; ++i) {
            res = max(res, left[i - 1] + right[i]);
        }
        return res;
    }

    int maxProfitIV(int k, vector<int> &prices) {
        int n = prices.size();
        vector<vector<int>> now(2, vector<int>(k + 1, 0)), prev(2, vector<int>(k + 1, 0));
        for (int j = 0; j <= k; ++j) {
            prev[1][j] = -prices[0];
            prev[0][j] = 0;
        }
        for (int i = 1; i < n; ++i) {
            now[1][k] = max(prev[0][k] - prices[i], prev[1][k]);
            now[0][k] = prev[0][k];
            for (int j = 0; j < k; ++j) {
                now[1][j] = max(prev[0][j] - prices[i], prev[1][j]);
                now[0][j] = max(prev[1][j + 1] + prices[i], prev[0][j]);
            }
            now.swap(prev);
        }
        return prev[0][0];
    }

    vector<int> maxSlidingWindow(vector<int> &nums, int k) {
        int n = nums.size();
        vector<int> res(n - k + 1);
        deque<int> q;
        auto push = [&](int index) -> void {
            while (!q.empty() && nums[q.back()] <= nums[index]) {
                q.pop_back();
            }
            q.push_back(index);
        };
        for (int i = 0; i < k; ++i) {
            push(i);
        }
        res[0] = nums[q[0]];
        for (int i = 1; i < res.size(); ++i) {
            while (!q.empty() && q.front() < i) {
                q.pop_front();
            }
            push(i + k - 1);
            res[i] = nums[q[0]];
        }
        return std::move(res);
    }

    vector<vector<int>> findDifference(vector<int> &nums1, vector<int> &nums2) {
        std::sort(nums1.begin(), nums1.end());
        std::sort(nums2.begin(), nums2.end());
        nums1.erase(std::unique(nums1.begin(), nums1.end()), nums1.end());
        nums2.erase(std::unique(nums2.begin(), nums2.end()), nums2.end());
        vector<vector<int>> res(2, vector<int>());
        for (auto i: nums1) {
            if (std::lower_bound(nums2.begin(), nums2.end(), i) == std::upper_bound(nums2.begin(), nums2.end(), i)) {
                res[0].push_back(i);
            }
        }
        for (auto i: nums2) {
            if (std::lower_bound(nums1.begin(), nums1.end(), i) == std::upper_bound(nums1.begin(), nums1.end(), i)) {
                res[1].push_back(i);
            }
        }
        return std::move(res);
    }

    bool isValid(const string &s) {
        stack<char> st;
        for (auto i: s) {
            if (i == 'c') {
                if (st.size() < 2) {
                    return false;
                }
                auto top1 = st.top();
                st.pop();
                auto top2 = st.top();
                st.pop();
                if (top1 == 'b' && top2 == 'a') {
                    continue;
                } else {
                    return false;
                }
            } else {
                st.push(i);
            }
        }
        return st.empty();
    }

    int findMaxValueOfEquation(vector<vector<int>> &points, int k) {
        int res = INT32_MIN;
        priority_queue<pair<int, int>> pq;
        for (auto &i: points) {
            while (!pq.empty() && pq.top().second + k < i[0]) {
                pq.pop();
            }
            if (pq.size()) {
                res = max(res, pq.top().first + i[0] + i[1]);
            }
            pq.emplace(i[1] - i[0], i[0]);
        }
        return res;
    }

    string predictPartyVictory(const string &senate) {
        queue<int> R, D;
        int n = senate.length();
        for (int i = 0; i < n; i++)
            (senate[i] == 'R') ? R.push(i) : D.push(i);
        while (R.size() && D.size()) {
            int r_index = R.front(), d_index = D.front();
            R.pop(), D.pop();
            (r_index < d_index) ? R.push(r_index + n) : D.push(d_index + n);
        }
        return (D.empty()) ? "Radiant" : "Dire";
    }

    int maxVowels(const string &s, int k) {
        auto f = [&](char c) -> bool {
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                return true;
            } else {
                return false;
            }
        };
        int num = 0;
        int res = 0;
        for (int i = 0; i < k; ++i) {
            if (f(s[i]))
                ++num;
        }
        res = max(res, num);
        for (int i = k; i < s.size(); ++i) {
            if (f(s[i - k]))
                --num;
            if (f(s[i]))
                ++num;
            res = max(res, num);
        }
        return res;
    }

    int numSubseq(vector<int> &nums, int target) {
        const long long int mod = 1e9 + 7;
        int n = nums.size();
        vector<int> pows(n, 1);
        for (int i = 1; i < pows.size(); ++i) {
            pows[i] = 2 * pows[i - 1] % mod;
        }
        long long int res = 0;
        std::sort(nums.begin(), nums.end());
        int l = 0;
        int r = n - 1;
        while (l <= r) {
            if (nums[l] + nums[r] > target) {
                r--;
            } else {
                res = (res + pows[r - l++]) % mod;
            }
        }
        return res;
    }
};

int main() {
    Solution s;
    vector<int> b = {2317, 3053, 2916, 6655, 6325, 3511, 4929, 3161, 5660, 2027, 2557, 2343, 2563, 5588, 6562, 5466,
                     5570, 5572, 314, 331, 922, 6504, 2559, 1793, 6504, 6086, 2563, 818, 3031, 2559, 2975, 2557, 2454,
                     4721, 2143, 5572, 3511, 2143, 3549, 331, 4674, 176, 2454, 5237, 6383, 1943, 527, 3370, 140, 88,
                     176, 1085, 2364, 4541, 2975, 1473, 2707, 4721, 5439, 3053, 64, 314, 5381, 5904, 6086, 3310, 3549,
                     4157, 166, 140, 2343, 5799, 203, 4934, 44, 4929, 2786, 44, 166, 5644, 6325, 5904, 5466};
    vector<int> a = {3697, 172, 5406, 5644, 5588, 4541, 2078, 172, 6492, 6152, 4545, 5660, 3310, 4525, 1971, 6655, 6562,
                     1793, 5938, 2317, 3459, 6889, 5799, 5237, 2027, 4545, 203, 3681, 6587, 3031, 3710, 6152, 578, 818,
                     3370, 5381, 88, 4525, 1971, 4157, 5439, 2078, 2590, 6712, 2786, 3681, 3618, 4396, 5268, 3459, 5570,
                     2916, 4396, 3525, 1085, 3618, 3525, 4934, 5406, 2707, 3995, 64, 5938, 3161, 2364, 2590, 527, 1943,
                     6587, 2184, 6383, 5268, 6492, 922, 3697, 578, 2184, 3710, 6889, 1473, 6712, 4674, 3995};
    cout << s.minCost(a, b) << endl;
    return 0;
}