use std::collections::VecDeque;

struct MKAverage {
    len: i32,
    capacity: i32,
    k: i32,
    stream: VecDeque<i32>,
    num_of_nums: [i32; 100001],
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MKAverage {
    fn new(m: i32, k: i32) -> Self {
        MKAverage {
            len: 0,
            capacity: m,
            k: k,
            stream: VecDeque::new(),
            num_of_nums: [0; 100001],
        }
    }

    fn add_element(&mut self, num: i32) {
        self.num_of_nums[num as usize] += 1;
        if self.len < self.capacity {
            self.len += 1;
        } else {
            self.num_of_nums[self.stream.pop_front().unwrap() as usize] -= 1;
        }
        self.stream.push_back(num);
    }

    fn calculate_mk_average(&self) -> i32 {
        if self.len < self.capacity {
            return -1;
        } else {
            let mut temp = 0;
            let end = self.capacity - self.k;
            let mut sum = 0;
            for i in 0..100001 {
                temp += self.num_of_nums[i];
                if temp <= self.k {
                    continue;
                } else if self.k + self.num_of_nums[i] > temp && temp >= end {
                    sum += (end - self.k) * i as i32;
                    break;
                } else if self.k + self.num_of_nums[i] > temp && temp < end {
                    sum += (temp - self.k) * i as i32;
                } else if self.k + self.num_of_nums[i] <= temp && temp <= end {
                    sum += self.num_of_nums[i] * i as i32;
                } else {
                    sum += (end + self.num_of_nums[i] - temp) * i as i32;
                    break;
                }
            }
            return sum / (self.capacity - 2 * self.k);
        }
    }
}
