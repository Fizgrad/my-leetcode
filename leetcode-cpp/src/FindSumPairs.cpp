
#include <unordered_map>
#include <vector>
using namespace std;

class FindSumPairs {
public:
    unordered_map<int, int> map2;
    vector<int> &nums2;
    vector<int> &nums1;


    FindSumPairs(vector<int> &nums1, vector<int> &nums2) : nums1(nums1), nums2(nums2) {
        for (auto i: nums2) {
            ++map2[i];
        }
    }

    void add(int index, int val) {
        --map2[nums2[index]];
        ++map2[nums2[index] + val];
        nums2[index] += val;
    }

    int count(int tot) {
        int res = 0;
        for (auto i: nums1) {
            res += map2[tot - i];
        }
        return res;
    }
};

/**
 * Your FindSumPairs object will be instantiated and called as such:
 * FindSumPairs* obj = new FindSumPairs(nums1, nums2);
 * obj->add(index,val);
 * int param_2 = obj->count(tot);
 */
int main() {
    return 0;
}