#include<algorithm>
#include <unordered_set>
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    static const long long int base = 200000 + 1;
    static const long long int sqbase = base * base;
    static const long long int bias = 100000;

    static int bisect_left(vector<int> &nums, int x, int a, int b) {
        int mid =(a + b) /2;
        while( mid < b && a < b){
            if (nums[mid] == x){
                return mid;
            }
            else if (nums[mid] > x){
                b = mid -1;
                mid =(a + b) /2;
            }
            else {
                a = mid +1;
                mid =(a + b) /2;
            }
        }
        return mid;
    }


    static vector<vector<int>> threeSum(vector<int> &nums) {
        int n = nums.size();
        unordered_set<long long int> res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; ++i) {
            if(i>0 && nums[i] == nums[i-1]) continue;
            for (int j = i + 1; j < n; ++j) {
                if(j>i+1 && nums[j] == nums[j-1]) continue;
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
};


int main(){
//    cout<<INT64_MAX<<endl;

    vector<int> a = {1,2,3,4,5,6,7,8,9};
    cout<< Solution::bisect_left(a,,0,a.size())<<endl;


}