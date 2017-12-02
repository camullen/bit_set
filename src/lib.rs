// Setup clippy
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![no_std]

// Use std for testing purposes
#[cfg(test)]
#[macro_use]
extern crate std;




extern crate generic_array;
extern crate typenum;

pub mod bitset;
pub mod iter;

pub use bitset::*;
pub use iter::*;
