#include <map>

using namespace std;

class MyCalendarTwo {
public:
    std::map<int, int> events;

    MyCalendarTwo() {
    }

    bool book(int start, int end) {
        events[start]++;
        events[end]--;
        int ongoing = 0;
        for (auto [time, count]: events) {
            ongoing += count;
            if (ongoing >= 3) {
                events[end]++;
                events[start]--;
                return false;
            }
        }
        return true;
    }
};

/**
 * Your MyCalendarTwo object will be instantiated and called as such:
 * MyCalendarTwo* obj = new MyCalendarTwo();
 * bool param_1 = obj->book(start,end);
 */