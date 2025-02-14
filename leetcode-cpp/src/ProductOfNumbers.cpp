#include <deque>
#include <vector>
class ProductOfNumbers {
public:
    std::vector<int> prefixProduct{1};
    bool zero = false;
    ProductOfNumbers() {}

    void add(int num) {
        if (num == 0) {
            zero = true;
            prefixProduct = {1};
        } else {
            prefixProduct.emplace_back(prefixProduct.back() * num);
        }
    }

    int getProduct(int k) {
        if (zero && k >= prefixProduct.size()) return 0;
        return prefixProduct.back() / *(prefixProduct.end() - k - 1);
    }
};

/**
* Your ProductOfNumbers object will be instantiated and called as such:
* ProductOfNumbers* obj = new ProductOfNumbers();
* obj->add(num);
* int param_2 = obj->getProduct(k);
*/

int main() {
    return 0;
}