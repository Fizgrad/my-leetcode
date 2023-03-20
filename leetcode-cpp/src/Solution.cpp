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


struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(NULL) {}
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

};

int main() {
    return 0;
}