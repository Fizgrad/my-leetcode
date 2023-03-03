//
// Created by David Chen on 3/3/23.
//
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    vector<string> getFolderNames(const vector<string>& names) {
        vector<string> res;
        res.reserve(names.size());
        unordered_map<string,int> hm;
        for(const auto& i : names){
            auto iter = hm.find(i);
            if(iter == hm.end()){
                res.push_back(i);
                ++iter->second;
            }else {
                int k  =  iter->second;
                string temp = i+'('+ to_string(k)+')';
                auto temp_iter = hm.find(temp);
                while(temp_iter != hm.end()){
                    ++k;
                    temp = i+'('+ to_string(k)+')';
                    temp_iter = hm.find(temp);
                }
                iter->second = k+1;
                hm[temp] = 1;
                res.push_back(temp);
            }
        }
        return res;
    }
};