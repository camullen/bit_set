use core::marker::PhantomData;
use bitset::BitValuable;

pub struct BitSetIterator<'a, I, T>
where
    I: Iterator<Item = &'a usize>,
    T: BitValuable,
{
    multi_obp_iter: MultiOneBitsPositionIterator<'a, I>,
    _phantom: PhantomData<T>,
}

impl<'a, I, T> Iterator for BitSetIterator<'a, I, T>
where
    I: Iterator<Item = &'a usize>,
    T: BitValuable,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.multi_obp_iter
            .next()
            .map(<T as BitValuable>::from_bit_value)
    }
}

impl<'a, I, T> BitSetIterator<'a, I, T>
where
    I: Iterator<Item = &'a usize>,
    T: BitValuable,
{
    pub fn new(iter: I) -> BitSetIterator<'a, I, T> {
        BitSetIterator {
            multi_obp_iter: MultiOneBitsPositionIterator::new(iter),
            _phantom: PhantomData,
        }
    }
}



pub struct ZeroBitsIntervalIterator(usize);

impl Iterator for ZeroBitsIntervalIterator {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let tz = self.0.trailing_zeros();
        self.0 >>= tz + 1;
        Some(tz as usize)
    }
}

pub struct OneBitsPositionIterator {
    zbi_iterator: ZeroBitsIntervalIterator,
    last_position: usize,
}

impl OneBitsPositionIterator {
    pub fn new(input: usize) -> OneBitsPositionIterator {
        OneBitsPositionIterator {
            zbi_iterator: ZeroBitsIntervalIterator(input),
            last_position: 0,
        }
    }
}

impl Iterator for OneBitsPositionIterator {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        self.zbi_iterator.next().map(|interval| {
            let ret_val = self.last_position + interval;
            self.last_position = ret_val + 1;
            ret_val
        })
    }
}


pub struct MultiOneBitsPositionIterator<'a, T>
where
    T: Iterator<Item = &'a usize>,
{
    iter: T,
    obp_iter: Option<OneBitsPositionIterator>,
    base: usize,
}

impl<'a, T> MultiOneBitsPositionIterator<'a, T>
where
    T: Iterator<Item = &'a usize>,
{
    pub fn new(iter: T) -> MultiOneBitsPositionIterator<'a, T> {
        MultiOneBitsPositionIterator {
            iter,
            obp_iter: None,
            base: 0,
        }
    }
}

impl<'a, T> Iterator for MultiOneBitsPositionIterator<'a, T>
where
    T: Iterator<Item = &'a usize>,
{
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        loop {
            if let Some(ref mut obp) = self.obp_iter {
                if let Some(x) = obp.by_ref().next() {
                    return Some(x + self.base - 64);
                }
            }
            match self.iter.next() {
                None => return None,
                Some(next_usize) => {
                    self.obp_iter = Some(OneBitsPositionIterator::new(*next_usize));
                    self.base += 64
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;
    use super::*;

    fn usize_from_intervals(intervals: &[usize]) -> usize {
        intervals
            .iter()
            .rev()
            .fold(0, |acc, &interval| ((acc << 1) | 1) << interval)
    }

    fn usize_from_positions(positions: &[usize]) -> usize {
        positions.iter().fold(0, |acc, &pos| acc | (1 << pos))
    }

    fn usize_vec_from_positions(positions: &[usize]) -> Vec<usize> {
        let max_pos = positions.iter().max().unwrap();
        let vec_size = (max_pos / 64) + 1;
        let mut usize_vec: Vec<usize> = Vec::with_capacity(vec_size);
        for _n in 0..vec_size {
            usize_vec.push(0);
        }
        for pos in positions.iter() {
            let index = pos / 64;
            let bitshift = pos % 64;
            let bitmask = 1 << bitshift;
            usize_vec[index] |= bitmask;
        }
        usize_vec
    }

    #[test]
    fn internal_tesst_usize_vec_from_positions() {
        let inputs = vec![15, 37, 78, 96, 107, 128, 131, 192, 255];
        let one_str = "10000000000000000000001000000000000000";
        let two_str = "10000000000100000000000000000100000000000000";
        let three_str = "1001";
        let four_str = "1000000000000000000000000000000000000000000000000000000000000001";
        let one = usize::from_str_radix(one_str, 2).unwrap();
        let two = usize::from_str_radix(two_str, 2).unwrap();
        let three = usize::from_str_radix(three_str, 2).unwrap();
        let four = usize::from_str_radix(four_str, 2).unwrap();
        let expected = vec![one, two, three, four];
        let actual = usize_vec_from_positions(&inputs);
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_zero_bits_interval_iterator() {
        let expected = vec![2, 14, 9, 21];
        let usize_input = usize_from_intervals(&expected);
        let actual: Vec<usize> = ZeroBitsIntervalIterator(usize_input).collect();

        assert_eq!(expected, actual)
    }

    #[test]
    fn test_zero_bits_interval_iterator_trailing_one() {
        let expected = vec![0, 2, 14, 9, 21];
        let usize_input = usize_from_intervals(&expected);
        let actual: Vec<usize> = ZeroBitsIntervalIterator(usize_input).collect();

        assert_eq!(expected, actual)
    }

    #[test]
    fn test_zero_bits_interval_iterator_zero_input() {
        let actual: Vec<usize> = ZeroBitsIntervalIterator(0).collect();
        assert!(actual.is_empty())
    }

    #[test]
    fn test_zero_bits_interval_iterator_repeating_ones() {
        let expected = vec![0, 0, 0, 2, 0, 14, 0, 0, 0, 9, 21];
        let usize_input = usize_from_intervals(&expected);
        let actual: Vec<usize> = ZeroBitsIntervalIterator(usize_input).collect();

        assert_eq!(expected, actual)
    }

    #[test]
    fn test_one_bits_position_iterator() {
        let expected = vec![2, 17, 27, 48];
        let usize_input = usize_from_positions(&expected);
        let actual: Vec<usize> = OneBitsPositionIterator::new(usize_input).collect();
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_one_bits_position_iterator_trailing_one() {
        let expected = vec![0, 1, 2, 17, 27, 48];
        let usize_input = usize_from_positions(&expected);
        let actual: Vec<usize> = OneBitsPositionIterator::new(usize_input).collect();
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_one_bits_position_iterator_zero_input() {
        let usize_input: usize = 0;
        let actual: Vec<usize> = OneBitsPositionIterator::new(usize_input).collect();
        assert!(actual.is_empty())
    }

}
