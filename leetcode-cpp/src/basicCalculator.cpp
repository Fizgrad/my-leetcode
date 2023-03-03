//
// Created by Fizgrad Chen on 2022/11/20.
//
//
//中缀转后缀C++代码实现(比较方便)
//1.遇到操作数：添加到后缀表达式中或直接输出
//2.栈空时：遇到运算符，直接入栈
//3.遇到左括号：将其入栈
//4.遇到右括号：执行出栈操作，输出到后缀表达式，直到弹出的是左括号
//注意：左括号不输出到后缀表达式
//5.遇到其他运算符：弹出所有优先级大于或等于该运算符的栈顶元素，然后将该运算符入栈
//6.将栈中剩余内容依次弹出后缀表达式

#include <iostream>
#include <stack>
#include <cctype>
using namespace std;

class Solution {
public:
    int calculate(string s) {
        stack<char> ops;
        string r;
        bool flag_p_d = false;
        long long int temp;
        for(auto i : s){
            if(isdigit(i)){
                if(!flag_p_d) {
                    flag_p_d = true;
                    temp = i-'0';
                }
                else {
                    temp *=10;
                    temp += i-'0';
                }
            }
            else {
                if(flag_p_d) {
                    flag_p_d = false;
                    r.append(to_string(temp));
                    r.push_back(',');
                    temp = 0;
                }
            }

            if(i =='('){
                ops.push(i);
            }
            else if(i == '+'||i=='-'){
                if(ops.empty()){
                    ops.push(i);
                }
                else {
                    while(!ops.empty()&&ops.top()!='('){
                        char op = ops.top();
                        ops.pop();
                        r.push_back(op);
                    }
                    ops.push(i);
                }
            }
            else if(i == ')'){
                while(!ops.empty()&&ops.top()!='('){
                    char op = ops.top();
                    ops.pop();
                    r.push_back(op);
                }
                if(ops.top()=='(')
                    ops.pop();
            }
        }
        if(flag_p_d) {
            flag_p_d = false;
            r.append(to_string(temp));
            r.push_back(',');
            temp = 0;
        }
        while(!ops.empty()){
            char op = ops.top();
            ops.pop();
            r.push_back(op);
        }
        cout<<r<<endl;
        stack<int> rnums;
        for(auto i:r){
            if(isdigit(i)){
                if(!flag_p_d) {
                    flag_p_d = true;
                    temp = i-'0';
                }
                else {
                    temp *=10;
                    temp += i-'0';
                }
            }
            else {
                if(flag_p_d) {
                    flag_p_d = false;
                    rnums.push(temp);
                    temp = 0;
                }
            }
            if(i == '+'){
                int num1 = rnums.top();
                rnums.pop();
                int num2 = rnums.top();
                rnums.pop();
                rnums.push(num1+num2);
            }
            else if(i == '-'){
                int num1 = rnums.top();
                rnums.pop();
                int num2 = rnums.top();
                rnums.pop();
                rnums.push(num2-num1);
            }
        }
        return rnums.top();

    }
};

int main(){
    Solution s ;
    cout<<s.calculate("2147483647")<<endl;


}
//
//
//class Solution {
//public:
//    int calculate(string s) {
//        int result=0;
//        int sum=0;
//        int sign=1;
//        stack<int>st;
//        int n=s.size();
//        for(int i=0;i<n;i++)
//        {
//            if(isdigit(s[i]))
//            {
//                sum=s[i] -'0';
//                while(i+1<n && isdigit(s[i+1]))
//                {
//                    sum = sum*10 + (s[i+1] -'0');
//                    i++;
//                }
//                result+=sum*sign;
//            }
//
//            else if(s[i] == '+')
//            {
//                sign=1;
//            }
//            else if (s[i] == '-')
//            {
//                sign=-1;
//            }
//            else if (s[i] == '(')
//            {
//                st.push(result);
//                st.push(sign);
//                result=0;
//                sign=1;
//            }
//            else if(s[i] == ')')
//            {
//                int xsign=st.top();
//                st.pop();
//                int xresult= st.top();
//                st.pop();
//                result=result*xsign + xresult;
//            }
//        }
//
//        return result;
//    }
//};