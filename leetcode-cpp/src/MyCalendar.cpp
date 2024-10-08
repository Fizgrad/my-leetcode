#include <iostream>
#include <map>
#include <utility>

using namespace std;


class MyCalendar {
public:
    map<int, int> intervals;

public:
    MyCalendar() {}

    bool book(int start, int end) {
        auto next = intervals.lower_bound(start);
        if (next != intervals.end() && next->first < end) {
            return false;
        }
        if (next != intervals.begin() && prev(next)->second > start) {
            return false;
        }
        intervals[start] = end;
        return true;
    }
};

int main() {
    return 0;
}
/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar* obj = new MyCalendar();
 * bool param_1 = obj->book(start,end);
 */