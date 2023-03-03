#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <cctype>
#include <unordered_set>
#include <cstring>
#include <stack>
#include <algorithm>
#include <unordered_map>
#include <sstream>
using namespace std;
//
//vector<int> a(int x){
//    vector<int> result;
//    for(int i =2;i<x;i++) {
//        if (x % i);
//        else {
//            result.push_back(i);
//        }
//    }
//    return result;
//}
//
//int count(int x){
//    vector<int> numbers = a(x);
//    if(numbers.empty())
//        return 1;
//    else {
//        int result =1;
//        for(auto i : numbers){
//                result += count(x/i) ;
//        }
//        return result;
//    }
//}
//
//int main() {
//    std::cout <<count(12)<< std::endl;
//    return 0;
//}
//
//string epoch(string input){
//    string result = "";
//    for(int i = 0;i<input.size();++i){
//        if(input[i] == '0')
//            result+="01";
//        if(input[i]=='1')
//            result+="10";
//
//    }
//    return result;
//}
//
//
//
//int main(){
//    string a = "0";
//    for(int i = 0;i<10;++i)
//    {
//        cout<<a<<endl;
//        a = epoch(a);
//    }
//    int k = 8;
//    cout<<exp2(log(k)/log(2))<<endl;
//}
//
//vector<string> fun1(string& a,int n){
//    vector<string> result_temp;
//    result_temp.push_back(a);
//    if(n == a.size())
//        return result_temp;
//    if(a[n] >='a'&&a[n]<='z' ){
//        a[n]=a[n] - 'a' + 'A';
//        result_temp.push_back(a);
//    }
//    else if(a[n]>='A'&&a[n]<='Z'){
//        a[n]=a[n] - 'A' + 'a';
//        result_temp.push_back(a);
//    }
//    return result_temp;
//}
//
//vector<string> fun2(vector<string>& a,int n){
//    vector<string> result;
//    if(n<a.begin()->size()) {
//        for (auto i: a) {
//            vector<string> aa = fun1(i, n);
//            for (auto j: aa) result.push_back(j);
//        }
//    }
//    return result;
//}
//int main(){
//    string input = "h1z1";
//    int i = 0;
//    vector<string> result;
//    result.push_back(input);
//    while(i<input.size()){
//        result = fun2(result,i);
//        ++i;
//    }
//    for(auto j : result)
//        cout<<j<<endl;
//}
//
//int solution(int n){
//    string s ="122112";
//    int sum = 6;
//    int j = 4;
//    while(sum < n){
//        int num = s[j] - '0';
//        char adder = !((s.back())-'1')+'1';
//        for(int i = 0;i< num;++i){
//            s.push_back(adder);
//        }
//        sum+= num;
//    }
//    int result = 0;
//    for(int i=0;i<n;++i ){
//        result+=(s[i]=='1');
//    }
//    return result;
//}
//
//int main(){
//
//    cout<<solution(6)<<endl;
//}
//
//
//int cutint(unsigned long long int a, bool negative) {
//    if(negative)
//        return (-a < INT32_MIN ? INT32_MIN : (unsigned  int ) a);
//    else {
//        return (a > INT32_MAX ? INT32_MAX : (unsigned  int ) a);
//    }
//}
//
//int myAtoi(string s) {
//    unsigned long long int temp = 0;
//    bool flag = false;
//    bool negative = false;
//    for (int i = 0; i < s.size(); ++i) {
//
//        if (s[i] == ' ') continue;
//
//        if (flag) {
//            if (isdigit(s[i])){
//                temp = temp * 10 + (s[i] - '0');
//                cout<<temp<<endl;
//            }
//
//            else return cutint(temp,negative);
//        } else {
//            if (isdigit(s[i])) {
//                flag = true;
//                temp = s[i] - '0';
//            } else if (s[i] == '+') {
//                flag = true;
//            } else if (s[i] == '-') {
//                flag = true;
//                negative = true;
//            } else return 0;
//        }
//    }
//    if (flag) {
//        if (negative)
//            return -cutint(temp,negative);
//        else return cutint(temp,negative);
//    }
//    return 0;
//}
//
//int main() {
//    cout<<atoi("     +0 123123")<<endl;
//
//}
//
//    bool splitArraySameAverage(vector<int>& nums) {
//        if (nums.size() == 1)
//            return false;
//        int sum = 0;
//        for(auto i : nums){
//            sum+=i;
//        }
//        bool flag = false;
//        for (int i = 1; i <= nums.size()/2; ++i) {
//            if (sum * i % nums.size() == 0) {
//               flag = true;
//            }
//        }
//        if(flag){
//            vector<unordered_set<int>> sums(nums.size()/2+1);
//            sums[0].insert(0);
//            for (int num: nums) {
//                for (int i = nums.size()/2; i >= 1; --i)
//                    for (const int t: sums[i-1])
//                        sums[i].insert(t + num);
//            }
//            for (int i = 1; i <= nums.size()/2; ++i)
//                if (sum*i%nums.size() == 0 && sums[i].find(sum*i/nums.size()) != sums[i].end()) return true;
//        }
//        return false;
//    }
//
//
//class Solution {
//public:
//    static string removeKdigits(string num, int k) {
//        if(num.size() == k|| num.empty())
//            return "0";
//        stack<int> s;
//        int count = 0;
//        s.push(num[0]-'0');
//        for(int i=1;i<num.size();++i){
//            int temp = num[i] -'0';
//            while(!s.empty()&&s.top()>temp&&count<k){
//                s.pop();
//                ++count;
//            }
//            s.push(temp);
//        }
//        while(count++<k){
//            s.pop();
//        }
//        vector<int> n;
//        string result;
//        while(!s.empty()){
//            n.push_back(s.top());
//            s.pop();
//        }
//        bool flag = false ;
//        for(auto i = n.rbegin();i!=n.rend();++i){
//            if (*i || flag){
//                flag =true;
//                result += (char)(*i + '0');
//            }
//        }
//        return result.empty()?"0":result;
//    }
//};
