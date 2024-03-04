//
// Created by David Chen on 5/31/23.
//
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

using namespace std;

class UndergroundSystem {
public:
    unordered_map<int, pair<int, int>> customers;
    unordered_map<string, int> string_tab;
    vector<unordered_map<int, pair<int, double>>> avg;

    int s2i(const string &input) {
        if (string_tab.count(input)) {
            return string_tab[input];
        } else {
            int id = string_tab.size();
            return string_tab[input] = id;
        }
    }

    UndergroundSystem() {
    }

    void checkIn(int id, string stationName, int t) {
        int from_station = s2i(stationName);
        customers[id] = {from_station, t};
        if (avg.size() <= from_station) {
            avg.resize(from_station + 1);
        }
    }

    void checkOut(int id, string stationName, int t) {
        auto [from, start] = customers[id];
        avg[from][s2i(stationName)].first += 1;
        avg[from][s2i(stationName)].second += (t - start);
    }

    double getAverageTime(string startStation, string endStation) {
        auto [times, sum] = avg[s2i(startStation)][s2i(endStation)];
        return sum / times;
    }
};

int main() { return 0; }
/**
 * Your UndergroundSystem object will be instantiated and called as such:
 * UndergroundSystem* obj = new UndergroundSystem();
 * obj->checkIn(id,stationName,t);
 * obj->checkOut(id,stationName,t);
 * double param_3 = obj->getAverageTime(startStation,endStation);
 */