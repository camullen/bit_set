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
    type MaxBitDiv64: ArrayLength<usize>;
    fn bit_value(&self) -> usize;
    fn from_bit_value(bit_value: usize) -> Self;
}

pub struct BitSet<T: BitValuable> {
    arr: GenericArray<usize, <T as BitValuable>::MaxBitDiv64>,
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

    pub fn max_allowed_bit_value(&self) -> usize {
        self.arr.len() * 64 - 1
    }

    fn check_bit_value(&self, bit_value: usize) {
        let max_allowed = self.max_allowed_bit_value();
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
        println!("Next iteration. Self = {:b}", self.0);
        if self.0 == 0 {
            println!("Returning none");
            return None;
        }
        let tz = self.0.trailing_zeros();
        println!("Trailing zeros = {}", tz);
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


struct MultiOneBitsPositionIterator<T>
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
    fn new(iter: T) -> MultiOneBitsPositionIterator<T> {
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


// struct MapOneBitsPositionIterator<T>
// where
// T: Iterator<Item = usize>,
// {
// iter: T,
// }

// impl<T> Iterator for MapOneBitsPositionIterator<T>
// where
// T: Iterator<Item = usize>,
// {
// type Item = OneBitsPositionIterator;
// fn next(&mut self) -> Option<OneBitsPositionIterator> {
// self.iter
// .next()
// .map(|next_usize| OneBitsPositionIterator::new(next_usize))
// }
// }


// struct MultiOneBitsPositionIteratorV<T>
// where T: Iterator<Item = usize>,
// {
// iter: T,
// obp_iterator: Option<OneBitsPositionIterator>,
// base: usize,
// initialized: bool,
// }

// impl<T> MultiOneBitsPositionIterator<T> {
// fn new(iter: T) -> MultiOneBitsPositionIterator<T> {
// MultiOneBitsPositionIterator {
// iter,
// obp_iterator: None,
// base: 0,
// }
// }
// }

// impl<T> Iterator for MultiOneBitsPositionIterator<T> {
// type Item = usize;
// fn next(&mut self) -> Option<usize> {
// self.obp_iterator = self.obp_iterator.or(self.iter.next().map(|usize_input| OneBitsPositionIterator::new(usize_input)));
// self.obp_iterator.and_then()
// }
// }



// pub struct BitSetIterator<'a, T: BitValuable + 'a> {
// bit_set: &'a BitSet<T>,
// curr_block: usize,
// curr_bit_pos: usize,
// }

// impl<'a, T: BitValuable + 'a> BitSetIterator<'a, T> {
// fn new(bit_set: &'a BitSet<T>) -> BitSetIterator<'a, T> {
// BitSetIterator {
// bit_set,
// curr_block: bit_set.arr[0],
// curr_bit_pos: 0,
// }
// }

// fn reset(&mut self) {
// self.curr_block = bit_set.arr[0];
// self.curr_bit_pos = 0;
// self.advance();
// }

// fn advance(&mut self) {
// if (self.curr_block & 1) != 0 {
// return;
// }
// self.advance_block();
// let trailing_zeroes = self.curr_block.trailing_zeroes();
// self.curr_bit_pos
// }

// fn advance_block(&mut self) -> bool {
// if self.curr_block != 0 {
// return;
// };

// let mut curr_index = self.curr_bit_pos / 64;
// let arr_len = self.bit_set.arr.len();

// while(self.curr_block == 0 && curr_index < arr_len) {
// curr_index += 1;
// self.curr_block = self.bit_set.arr[curr_index];
// }
// self.curr_bit_pos = curr_index * 64;
// }
// }

// impl<'a, T: BitValuable + 'a> Iterator for BitSetIterator<'a, T> {
// type Item = T;
// fn next(&mut self) -> Option<T> {
// if(self.curr_block_index >= bit_set.arr.len()) {
// self.curr_block_index = 0;
// self.curr_block = bit_set.arr[0];
// self.curr_bit_pos = 0;
// return None;
// }
// let bit_value = self.curr_bit_pos
// }
// }

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;
    use super::*;

    fn usize_from_intervals(intervals: &[usize]) -> usize {
        intervals
            .iter()
            .rev()
            .fold(0, |acc, &interval| ((acc << 1) | 1) << interval);
    }

    #[test]
    fn test_zero_bits_interval_iterator() {
        let expected = vec![2, 14, 9, 21];
        let usize_input: usize = expected
            .iter()
            .rev()
            .fold(0, |acc, &interval| ((acc << 1) | 1) << interval);
        let actual: Vec<usize> = ZeroBitsIntervalIterator(usize_input).collect();

        assert_eq!(expected, actual)
    }

    #[test]
    fn test_zero_bits_interval_iterator_trailing_one() {
        let expected = vec![0, 2, 14, 9, 21];
        let usize_input: usize = expected
            .iter()
            .rev()
            .fold(0, |acc, &interval| ((acc << 1) | 1) << interval);
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
        let usize_input: usize = expected
            .iter()
            .rev()
            .fold(0, |acc, &interval| ((acc << 1) | 1) << interval);
        let actual: Vec<usize> = ZeroBitsIntervalIterator(usize_input).collect();

        assert_eq!(expected, actual)
    }

    #[test]
    fn test_one_bits_position_iterator() {
        let expected = vec![2, 17, 27, 48];
        let usize_input: usize = expected.iter().fold(0, |acc, &pos| acc | (1 << pos));
        let actual: Vec<usize> = OneBitsPositionIterator::new(usize_input).collect();
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_one_bits_position_iterator_trailing_one() {
        let expected = vec![0, 1, 2, 17, 27, 48];
        let usize_input: usize = expected.iter().fold(0, |acc, &pos| acc | (1 << pos));
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
