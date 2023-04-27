//
// Created by David Chen on 4/27/23.
//
#include <iostream>
#include <vector>

using namespace std;

class ATM {
public:
    vector<long long int> banknotesCount;
    constexpr static int num[5] = {20, 50, 100, 200, 500};

    ATM() : banknotesCount(5, 0) {

    }

    void deposit(vector<int> banknotesCount) {
        for (int i = 0; i < banknotesCount.size(); ++i) {
            this->banknotesCount[i] += banknotesCount[i];
        }
    }

    vector<int> withdraw(long long int amount) {
        vector<int> temp(5, 0);
        for (int i = 4; i >= 0; --i) {
            if (!banknotesCount[i]) {
                continue;
            }
            if (amount >= num[i]) {
                amount -= ((temp[i] = min((amount / num[i]), banknotesCount[i])) * num[i]);
            }
        }
        if (amount) {
            return {-1};
        }
        for (int i = 4; i >= 0; --i) {
            banknotesCount[i] -= temp[i];
        }
        return std::move(temp);
    }
};

/**
 * Your ATM object will be instantiated and called as such:
 * ATM* obj = new ATM();
 * obj->deposit(banknotesCount);
 * vector<int> param_2 = obj->withdraw(amount);
 */