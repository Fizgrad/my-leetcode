#include <set>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

vector<string> foodsStrings;
vector<string> cuisinesStrings;
unordered_map<string, int> foodsStringIndex;
unordered_map<string, int> cuisinesStringIndex;

unordered_map<int, int> foods2ratings;
unordered_map<int, int> foods2cuisines;

class cmp {
public:
    bool operator()(int a, int b) const {
        return foods2ratings[a] > foods2ratings[b] || (foods2ratings[a] == foods2ratings[b] && foodsStrings[a] < foodsStrings[b]);
    }
};

int toFoodsStringIndex(const string &a) {
    auto iter = foodsStringIndex.find(a);
    if (iter == foodsStringIndex.end()) {
        auto nextIndex = foodsStrings.size();
        foodsStringIndex[a] = nextIndex;
        foodsStrings.emplace_back(a);
        return nextIndex;
    } else {
        return iter->second;
    }
}


int toCuisinesStringIndex(const string &a) {
    auto iter = cuisinesStringIndex.find(a);
    if (iter == cuisinesStringIndex.end()) {
        auto nextIndex = cuisinesStrings.size();
        cuisinesStringIndex[a] = nextIndex;
        cuisinesStrings.emplace_back(a);
        return nextIndex;
    } else {
        return iter->second;
    }
}

unordered_map<int, set<int, cmp>> cuisines2foods;

class FoodRatings {
public:
    FoodRatings(vector<string> &foods, vector<string> &cuisines, vector<int> &ratings) {
        foods2ratings.clear();
        foods2cuisines.clear();
        cuisines2foods.clear();
        cuisinesStringIndex.clear();
        cuisinesStrings.clear();
        foodsStringIndex.clear();
        foodsStrings.clear();
        int n = foods.size();
        foodsStrings.reserve(n);
        // vector<int> indices(n);
        // for (int i = 0; i < n; ++i) {
        //     auto foodIndex = toFoodsStringIndex(foods[i]);
        //     indices[i] = foodIndex;
        // }
        // std::sort(indices.begin(), indices.end(), [&](auto &a, auto &b) {
        //     return foodsStrings[a] < foodsStrings[b];
        // });
        // for (int i = 0; i < n; ++i) {
        //     foodsStringIndex[foodsStrings[indices[i]]] = i;
        // }
        // std::sort(foodsStrings.begin(), foodsStrings.end());

        for (int i = 0; i < n; ++i) {
            auto foodIndex = toFoodsStringIndex(foods[i]);
            foods2ratings[foodIndex] = ratings[i];
            foods2cuisines[foodIndex] = toCuisinesStringIndex(cuisines[i]);
        }
        for (int i = 0; i < n; ++i) {
            cuisines2foods[toCuisinesStringIndex(cuisines[i])].insert(toFoodsStringIndex(foods[i]));
        }
    }

    void changeRating(const string &food, int newRating) {
        auto foodIndex = toFoodsStringIndex(food);
        auto &cuisineIndex = foods2cuisines[foodIndex];
        cuisines2foods[cuisineIndex].erase(foodIndex);
        foods2ratings[foodIndex] = newRating;
        cuisines2foods[cuisineIndex].insert(foodIndex);
    }

    string highestRated(const string &cuisine) {
        return foodsStrings[*cuisines2foods[toCuisinesStringIndex(cuisine)].begin()];
    }
};

/**
 * Your FoodRatings object will be instantiated and called as such:
 * FoodRatings* obj = new FoodRatings(foods, cuisines, ratings);
 * obj->changeRating(food,newRating);
 * string param_2 = obj->highestRated(cuisine);
 */