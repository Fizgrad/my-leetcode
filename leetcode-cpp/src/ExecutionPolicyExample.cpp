#include <algorithm>
#include <chrono>
#include <cstdint>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

void measureSimpleOperation(auto &&policy, std::vector<uint64_t> v) {
    auto start = std::chrono::steady_clock::now();
    std::for_each(policy, v.begin(), v.end(), [](auto i) {
        auto b = i >> 1;
    });
    auto finish = std::chrono::steady_clock::now();
    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
              << " ms\n";
}


void measureComplexOperation(auto &&policy, std::vector<uint64_t> v) {
    auto start = std::chrono::steady_clock::now();
    std::for_each(policy, v.begin(), v.end(), [](auto i) {
        int64_t sum = 0;
        for (int k = 0; k < i; ++k) {
            sum += k;
        }
    });
    auto finish = std::chrono::steady_clock::now();
    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
              << " ms\n";
}


int main() {
    std::vector<uint64_t> v(20'000'000);
    std::mt19937 gen{std::random_device{}()};
    std::ranges::generate(v, gen);

    std::cout << "Sequential policy:\n";
    // 指定算法的执行必须是顺序的，即不允许并行化
    measureSimpleOperation(std::execution::seq, v);

    std::cout << "Parallel policy:\n";
    // 允许算法的执行并行化
    measureSimpleOperation(std::execution::par, v);

    std::cout << "Parallel unsequenced policy:\n";
    // 允许算法的执行并行化、向量化，或在线程间迁移
    measureSimpleOperation(std::execution::par_unseq, v);

    std::cout << "Unsequenced policy:\n";
    // 允许算法的执行向量化，但不允许并行化（C++20 新增）
    measureSimpleOperation(std::execution::unseq, v);

    v = std::vector<uint64_t>(40'000'000);
    std::cout << "Sequential policy:\n";
    // 指定算法的执行必须是顺序的，即不允许并行化
    measureComplexOperation(std::execution::seq, v);

    std::cout << "Parallel policy:\n";
    // 允许算法的执行并行化
    measureComplexOperation(std::execution::par, v);

    std::cout << "Parallel unsequenced policy:\n";
    // 允许算法的执行并行化、向量化，或在线程间迁移
    measureComplexOperation(std::execution::par_unseq, v);

    std::cout << "Unsequenced policy:\n";
    // 允许算法的执行向量化，但不允许并行化（C++20 新增）
    measureComplexOperation(std::execution::unseq, v);
    return 0;
}
