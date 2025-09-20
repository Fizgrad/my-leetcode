#include <cctype>
#include <string>
#include <vector>
using namespace std;

class Spreadsheet {
public:
    vector<vector<int>> data;

    Spreadsheet(int rows) {
        data.resize('Z' - 'A' + 1, vector<int>(rows + 1, 0));
    }

    int *getCellAddress(const string &cell) {
        return &data[cell.front() - 'A'][stoi(cell.substr(1))];
    }

    int getCell(const string &cell) {
        return *getCellAddress(cell);
    }

    void setCell(const string &cell, int value) {
        *getCellAddress(cell) = value;
    }

    void resetCell(const string &cell) {
        setCell(cell, 0);
    }

    int getValue(const string &formula) {
        auto plus = formula.find('+');
        string operand1 = formula.substr(1, plus - 1);
        string operand2 = formula.substr(plus + 1);
        int res = 0;
        if (isalpha(operand1.front())) {
            res += getCell(operand1);
        } else {
            res += stoi(operand1);
        }
        if (isalpha(operand2.front())) {
            res += getCell(operand2);
        } else {
            res += stoi(operand2);
        }
        return res;
    }
};

/**
 * Your Spreadsheet object will be instantiated and called as such:
 * Spreadsheet* obj = new Spreadsheet(rows);
 * obj->setCell(cell,value);
 * obj->resetCell(cell);
 * int param_3 = obj->getValue(formula);
 */