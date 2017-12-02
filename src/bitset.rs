use core;
use core::iter::FromIterator;
pub use typenum::consts::*;
pub use typenum::marker_traits::Unsigned;

use generic_array::{ArrayLength, GenericArray};
use iter::BitSetIterator;

pub trait BitValuable {
    type MaxBit: Unsigned;
    type ArrLen: ArrayLength<usize>;
    fn bit_value(&self) -> usize;
    fn from_bit_value(bit_value: usize) -> Self;
}

impl BitValuable for u8 {
    type MaxBit = U255;
    type ArrLen = U4;

    fn bit_value(&self) -> usize {
        *self as usize
    }

    fn from_bit_value(bit_value: usize) -> u8 {
        bit_value as u8
    }
}

impl BitValuable for i8 {
    type MaxBit = U255;
    type ArrLen = U4;

    fn bit_value(&self) -> usize {
        *self as usize
    }

    fn from_bit_value(bit_value: usize) -> i8 {
        bit_value as i8
    }
}


#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
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

    pub fn symmetric_difference(&self, other: &BitSet<T>) -> BitSet<T> {
        let diff_arr = self.arr.zip_ref(&other.arr, |s, o| *s ^ *o);
        let diff_len: usize = diff_arr.iter().map(|b| b.count_ones() as usize).sum();
        BitSet {
            arr: diff_arr,
            len: diff_len,
        }
    }

    pub fn difference(&self, other: &BitSet<T>) -> BitSet<T> {
        let diff_arr = self.arr.zip_ref(&other.arr, |s, o| (*s & *o) ^ *s);
        let diff_len: usize = diff_arr.iter().map(|b| b.count_ones() as usize).sum();
        BitSet {
            arr: diff_arr,
            len: diff_len,
        }
    }

    pub fn intersection(&self, other: &BitSet<T>) -> BitSet<T> {
        let diff_arr = self.arr.zip_ref(&other.arr, |s, o| *s & *o);
        let diff_len: usize = diff_arr.iter().map(|b| b.count_ones() as usize).sum();
        BitSet {
            arr: diff_arr,
            len: diff_len,
        }
    }

    pub fn union(&self, other: &BitSet<T>) -> BitSet<T> {
        let diff_arr = self.arr.zip_ref(&other.arr, |s, o| *s | *o);
        let diff_len: usize = diff_arr.iter().map(|b| b.count_ones() as usize).sum();
        BitSet {
            arr: diff_arr,
            len: diff_len,
        }
    }

    pub fn is_disjoint(&self, other: &BitSet<T>) -> bool {
        self.arr
            .iter()
            .zip(other.arr.iter())
            .all(|(s, o)| (*s & *o) == 0)
    }

    pub fn is_subset(&self, other: &BitSet<T>) -> bool {
        self.arr
            .iter()
            .zip(other.arr.iter())
            .all(|(s, o)| (*s & *o) == *s)
    }

    pub fn is_superset(&self, other: &BitSet<T>) -> bool {
        self.arr
            .iter()
            .zip(other.arr.iter())
            .all(|(s, o)| (*s & *o) == *o)
    }

    pub fn iter(&self) -> BitSetIterator<core::slice::Iter<usize>, T> {
        BitSetIterator::new(self.arr.iter())
    }
    fn contains_entry(&self, entry: &BitEntry) -> bool {
        (self.arr[entry.index] & entry.bitmask) != 0
    }

    fn insert_entry(&mut self, entry: &BitEntry) {
        self.arr[entry.index] |= entry.bitmask;
        self.len += 1;
    }

    fn remove_entry(&mut self, entry: &BitEntry) {
        self.arr[entry.index] &= !entry.bitmask;
        self.len -= 1;
    }

    pub fn max_allowed_bit_value() -> usize {
        <T::MaxBit as Unsigned>::to_usize()
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

impl<'a, T> IntoIterator for &'a BitSet<T>
where
    T: BitValuable,
{
    type Item = T;
    type IntoIter = BitSetIterator<'a, core::slice::Iter<'a, usize>, T>;
    fn into_iter(self) -> BitSetIterator<'a, core::slice::Iter<'a, usize>, T> {
        self.iter()
    }
}

impl<T> FromIterator<T> for BitSet<T>
where
    T: BitValuable,
{
    fn from_iter<I>(iter: I) -> BitSet<T>
    where
        I: IntoIterator<Item = T>,
    {
        let mut set = BitSet::new();
        for i in iter {
            set.insert(&i);
        }
        set
    }
}

impl<'a, T> FromIterator<&'a T> for BitSet<T>
where
    T: BitValuable + 'a,
{
    fn from_iter<I>(iter: I) -> BitSet<T>
    where
        I: IntoIterator<Item = &'a T>,
    {
        let mut set = BitSet::new();
        for i in iter {
            set.insert(i);
        }
        set
    }
}


impl<T> Extend<T> for BitSet<T>
where
    T: BitValuable,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for i in iter {
            self.insert(&i);
        }
    }
}


impl<'a, T> Extend<&'a T> for BitSet<T>
where
    T: BitValuable + 'a,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        for i in iter {
            self.insert(i);
        }
    }
}

impl<T> Default for BitSet<T>
where
    T: BitValuable,
{
    fn default() -> BitSet<T> {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use std::prelude::v1::*;
    use super::*;

    #[test]
    fn bitset_smoke_test() {
        let insert: u8 = 120;
        let mut bit_set = BitSet::new();
        assert!(bit_set.is_empty());
        assert_eq!(0, bit_set.len());
        assert!(!bit_set.contains(&insert));

        let insert_res = bit_set.insert(&insert);

        assert!(insert_res);
        assert!(!bit_set.is_empty());
        assert_eq!(1, bit_set.len());
        assert!(bit_set.contains(&insert));

        let dup_insert_res = bit_set.insert(&insert);

        assert!(!dup_insert_res);
        assert!(!bit_set.is_empty());
        assert_eq!(1, bit_set.len());
        assert!(bit_set.contains(&insert));

        let remove_res = bit_set.remove(&insert);
        assert!(remove_res);
        assert!(bit_set.is_empty());
        assert_eq!(0, bit_set.len());
        assert!(!bit_set.contains(&insert));

        let dup_remove_res = bit_set.remove(&insert);
        assert!(!dup_remove_res);
        assert!(bit_set.is_empty());
        assert_eq!(0, bit_set.len());
        assert!(!bit_set.contains(&insert));
    }

}
