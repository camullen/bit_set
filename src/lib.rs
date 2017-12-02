// #![no_std]

// #[cfg(test)]
// #[macro_use]
// extern crate std;

#[macro_use]
extern crate typenum;

#[macro_use]
extern crate generic_array;

pub use typenum::consts::*;
pub use typenum::marker_traits::Unsigned;

use typenum::uint::UInt;
use generic_array::{ArrayLength, GenericArray};

pub trait BitValuable {
    type MaxBit: Unsigned;
    type ArrLen: ArrayLength<usize>;
    fn bit_value(&self) -> usize;
    fn from_bit_value(bit_value: usize) -> Self;
}

pub struct BitSet<T: BitValuable> {
    arr: GenericArray<usize, <T as BitValuable>::ArrLen>,
    len: usize,
}

struct BitEntry {
    index: usize,
    bitmask: usize,
}

impl BitEntry {
    fn new<T: BitValuable>(bit_value: &T) -> BitEntry {
        let bit_value = bit_value.bit_value();
        BitEntry {
            index: bit_value / 64,
            bitmask: 1 << (bit_value % 64),
        }
    }
}

impl<T: BitValuable> BitSet<T> {
    pub fn new() -> BitSet<T> {
        BitSet {
            arr: GenericArray::default(),
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        for block in self.arr.iter_mut() {
            *block = 0
        }
        self.len = 0;
    }

    pub fn insert(&mut self, value: &T) -> bool {
        self.check_bit_value(value.bit_value());
        let entry = BitEntry::new(value);
        if self.contains_entry(&entry) {
            return false;
        };
        self.insert_entry(&entry);
        true
    }

    pub fn contains(&self, value: &T) -> bool {
        self.check_bit_value(value.bit_value());
        let entry = BitEntry::new(value);
        self.contains_entry(&entry)
    }

    pub fn remove(&mut self, value: &T) -> bool {
        self.check_bit_value(value.bit_value());
        let entry = BitEntry::new(value);
        if !self.contains_entry(&entry) {
            return false;
        };
        self.remove_entry(&entry);
        true
    }

    fn contains_entry(&self, entry: &BitEntry) -> bool {
        (self.arr[entry.index] & entry.bitmask) != 0
    }

    fn insert_entry(&mut self, entry: &BitEntry) {
        self.arr[entry.index] |= entry.bitmask;
        self.len += 1;
    }

    fn remove_entry(&mut self, entry: &BitEntry) {
        self.arr[entry.index] &= (!entry.bitmask);
        self.len -= 1;
    }

    pub fn max_allowed_bit_value() -> usize {
        <T::MaxBit as Unsigned>::to_usize()
        // self.arr.len() * 64 - 1
    }

    fn check_bit_value(&self, bit_value: usize) {
        let max_allowed = Self::max_allowed_bit_value();
        if bit_value > max_allowed {
            panic!(
                "Tried to perform operation with value having bit_value of {}. Max bit_value is {}",
                bit_value,
                max_allowed
            )
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
        self.0 = self.0 >> (tz + 1);
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


pub struct MultiOneBitsPositionIterator<T>
where
    T: Iterator<Item = usize>,
{
    iter: T,
    obp_iter: Option<OneBitsPositionIterator>,
    base: usize,
}

impl<T> MultiOneBitsPositionIterator<T>
where
    T: Iterator<Item = usize>,
{
    pub fn new(iter: T) -> MultiOneBitsPositionIterator<T> {
        MultiOneBitsPositionIterator {
            iter,
            obp_iter: None,
            base: 0,
        }
    }
}

impl<T> Iterator for MultiOneBitsPositionIterator<T>
where
    T: Iterator<Item = usize>,
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
                    self.obp_iter = Some(OneBitsPositionIterator::new(next_usize));
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
        for n in 0..vec_size {
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
