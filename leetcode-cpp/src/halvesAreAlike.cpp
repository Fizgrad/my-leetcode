//
// Created by Fizgrad Chen on 2022/12/1.
//
#include <iostream>
#include <string>
using  namespace std;
class Solution {
public:
    bool halvesAreAlike(const string& s) {
        int size = s.size();
        int a = 0;
        int b = 0;
        for(int i=0;i<size/2;++i){
            if(s[i] == 'a' ||s[i] == 'e' ||s[i] == 'i' ||s[i] == 'o' ||s[i] == 'u' ||s[i] == 'A' ||s[i] == 'E' ||s[i] == 'I' ||s[i] == 'O' ||s[i] == 'U')
            {
                ++a;
            }
        }
        for(int i=size/2;i<size;++i){
            if(s[i] == 'a' ||s[i] == 'e' ||s[i] == 'i' ||s[i] == 'o' ||s[i] == 'u' ||s[i] == 'A' ||s[i] == 'E' ||s[i] == 'I' ||s[i] == 'O' ||s[i] == 'U')
            {
                ++b;
            }
        }
        return a==b;
    }
};