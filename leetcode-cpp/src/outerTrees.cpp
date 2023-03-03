#include <vector>
#include <iostream>
#include <stack>
#include <set>
using namespace std;


class Solution {
public:
    static int crossProduct(vector<int> p1,vector<int> p2,vector<int> p3) {
        int a = p2[0] - p1[0];
        int b = p2[1] - p1[1];
        int c = p3[0] - p1[0];
        int d = p3[1] - p1[1];
        return a * d - b * c;
    }

    static vector<vector<int>>  constructHalfHull(vector<vector<int>>& trees){
        vector<vector<int>> q;
        for(int i = 0;i<trees.size();++i){
            while(q.size()>=2&& crossProduct(*(q.end()-2),*(q.end()-1),trees[i])>0){
                q.pop_back();
            }
            q.push_back(trees[i]);
        }
        return q;
    }

    static vector<vector<int>> outerTrees(vector<vector<int>>& trees) {
        if(trees.size()<=3){
            return trees;
        }
        sort(trees.begin(),trees.end());

        auto m = constructHalfHull(trees);
        vector<vector<int>> temp ;
        for(auto i = trees.rbegin();i != trees.rend();++i){
            temp.push_back(*i);
        }
        auto n = constructHalfHull(temp);
        set<vector<int>> r;
        for(auto i : m){
            r.insert(i);
        }
        for(auto i : n){
            r.insert(i);
        }
        vector<vector<int>> q;
        for(auto i : r){
            q.push_back(i);
        }

        return q;
    }
};



int main(){


    vector<vector<int>> a ={{0,1},{0,0},{1,0},{-1,0},{0,-1}};
    auto r = Solution::outerTrees(a);
    for(auto i:r){
        for(auto j : i){
            cout<<j <<" ";
        }
        cout<<endl;
    }
    return 0;
}