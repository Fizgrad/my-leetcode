struct SummaryRanges {
    nums: Vec<bool>,
}

impl SummaryRanges {
    fn new() -> Self {
        SummaryRanges {
            nums: std::vec![false;10001],
        }
    }

    fn add_num(&mut self, value: i32) {
        self.nums[value as usize] = true;
    }

    fn get_intervals(&self) -> Vec<Vec<i32>> {
        let mut temp = 0;
        let mut flag = false;
        let mut res = std::vec![];
        for (i, &exists) in self.nums.iter().enumerate() {
            if exists {
                if !flag {
                    flag = true;
                    temp = i;
                } else {
                    continue;
                }
            } else {
                if flag {
                    res.push(std::vec![temp as i32, (i - 1) as i32]);
                    flag = false;
                }
            }
        }
        if flag {
            res.push(std::vec![temp as i32, 10000 as i32]);
            flag = false;
        }
        return res;
    }
}
