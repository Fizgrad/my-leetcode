//
// Created by David Chen on 4/24/23.
//
#include<iostream>
#include <vector>

using namespace std;

class SegmentTree {
public:
    vector<int> d;
    vector<int> &a;

    SegmentTree(vector<int> &a) : a(a), d(a.size() * 4, 0) {
        build(0, a.size() - 1, 1);
    }

    void build(int s, int t, int p) {
        // 对 [s,t] 区间建立线段树,当前根的编号为 p
        if (s == t) {
            d[p] = a[s];
            return;
        }
        int m = s + ((t - s) >> 1);
        // 移位运算符的优先级小于加减法，所以加上括号
        // 如果写成 (s + t) >> 1 可能会超出 int 范围
        build(s, m, p * 2), build(m + 1, t, p * 2 + 1);
        // 递归对左右区间建树
        d[p] = d[p * 2] + d[(p * 2) + 1];
    }

    int getsum(int l, int r) {
        return getsum(l, r, 0, a.size() - 1, 1);
    }

    int getsum(int l, int r, int s, int t, int p) {
        // [l, r] 为查询区间, [s, t] 为当前节点包含的区间, p 为当前节点的编号
        if (l <= s && t <= r)
            return d[p];  // 当前区间为询问区间的子集时直接返回当前区间的和
        int m = s + ((t - s) >> 1), sum = 0;
        if (l <= m) sum += getsum(l, r, s, m, p * 2);
        // 如果左儿子代表的区间 [s, m] 与询问区间有交集, 则递归查询左儿子
        if (r > m) sum += getsum(l, r, m + 1, t, p * 2 + 1);
        // 如果右儿子代表的区间 [m + 1, t] 与询问区间有交集, 则递归查询右儿子
        return sum;
    }

    void update(int index, int diff) {
        update(index, diff, 0, a.size() - 1, 1);
    }

    void update(int index, int diff, int l, int r, int p) {
        if (l <= index && r >= index) {
            d[p] += diff;
            int m = (l + r) >> 1;
            if (r != l) {
                if (m + 1 <= index)
                    update(index, diff, m + 1, r, 2 * p + 1);
                if (index <= m)
                    update(index, diff, l, m, 2 * p);
            }
        }
    }
};

class NumArray {
public:
    SegmentTree st;

    NumArray(vector<int> &nums) : st(nums) {
    }

    void update(int index, int val) {
        st.update(index, val - st.a[index]);
        st.a[index] = val;
    }

    int sumRange(int left, int right) {
        return st.getsum(left, right);
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(index,val);
 * int param_2 = obj->sumRange(left,right);
 */