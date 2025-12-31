#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
class DynamicBitset {
public:
    using size_type = std::size_t;

    // 1) 默认构造：size=0
    DynamicBitset() = default;

    // 2) 指定大小，默认全 0
    explicit DynamicBitset(size_type size)
        : nbits_(size), words_((size + 63) / 64, 0ULL) {}

    // 3) 指定大小 + 初始值（fill=0 全0，fill=1 全1）
    DynamicBitset(size_type size, int fill01)
        : nbits_(size), words_((size + 63) / 64, (fill01 != 0) ? ~0ULL : 0ULL) {
        mask_tail_bits_();
    }

    // 拷贝构造/拷贝赋值/移动构造/移动赋值：默认就够用
    DynamicBitset(const DynamicBitset &) = default;
    DynamicBitset &operator=(const DynamicBitset &) = default;
    DynamicBitset(DynamicBitset &&) noexcept = default;
    DynamicBitset &operator=(DynamicBitset &&) noexcept = default;

    // 基本信息
    size_type size() const noexcept { return nbits_; }
    bool empty() const noexcept { return nbits_ == 0; }

    // 清空：把所有位变为 0（size 不变）
    void clear() noexcept {
        for (auto &w: words_) w = 0ULL;
    }

    // any：是否存在至少一个 1
    bool any() const noexcept {
        for (auto w: words_) {
            if (w != 0ULL) return true;
        }
        return false;
    }

    // test：读取某一位
    bool test(size_type pos) const {
        check_pos_(pos);
        auto [wi, bi] = split_(pos);
        return (words_[wi] >> bi) & 1ULL;
    }

    // set：设置某一位为 1
    void set(size_type pos) {
        check_pos_(pos);
        auto [wi, bi] = split_(pos);
        words_[wi] |= (1ULL << bi);
    }

    // reset：设置某一位为 0
    void reset(size_type pos) {
        check_pos_(pos);
        auto [wi, bi] = split_(pos);
        words_[wi] &= ~(1ULL << bi);
    }

    bool all() const noexcept {
        // 空集合：按“真空真”(vacuous truth) 处理，返回 true
        if (nbits_ == 0) return true;
        if (words_.empty()) return true;

        // 前面的完整 word 必须全是 1
        for (size_type i = 0; i + 1 < words_.size(); ++i) {
            if (words_[i] != ~0ULL) return false;
        }

        // 最后一个 word：只要求有效位为 1
        const unsigned rem = static_cast<unsigned>(nbits_ % WORD_BITS);
        const std::uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
        return words_.back() == mask;
    }

    // （可选）设置所有位：全 1 / 全 0
    void set_all() noexcept {
        for (auto &w: words_) w = ~0ULL;
        mask_tail_bits_();
    }
    void reset_all() noexcept { clear(); }
    // ===== 位或运算符重载 =====

    // a |= b
    DynamicBitset &operator|=(const DynamicBitset &rhs) {
        require_same_size_(rhs);
        for (size_type i = 0; i < words_.size(); ++i) {
            words_[i] |= rhs.words_[i];
        }
        // 理论上尾部位不会被置脏（rhs 也应是干净的），但防御一下没坏处
        mask_tail_bits_();
        return *this;
    }

    // c = a | b
    friend DynamicBitset operator|(DynamicBitset lhs, const DynamicBitset &rhs) {
        lhs |= rhs;
        return lhs;
    }

private:
    size_type nbits_ = 0;
    std::vector<std::uint64_t> words_;

    static constexpr size_type WORD_BITS = 64;

    static std::pair<size_type, unsigned> split_(size_type pos) noexcept {
        return {pos / WORD_BITS, static_cast<unsigned>(pos % WORD_BITS)};
    }

    void check_pos_(size_type pos) const {
        if (pos >= nbits_) throw std::out_of_range("DynamicBitset: bit position out of range");
    }

    void require_same_size_(const DynamicBitset &rhs) const {
        if (nbits_ != rhs.nbits_) {
            throw std::invalid_argument("DynamicBitset: size mismatch for bitwise OR");
        }
    }

    void mask_tail_bits_() noexcept {
        if (nbits_ == 0 || words_.empty()) return;
        const unsigned rem = static_cast<unsigned>(nbits_ % WORD_BITS);
        if (rem == 0) return;
        const std::uint64_t mask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
        words_.back() &= mask;
    }
};

int main() {
    return 0;
}