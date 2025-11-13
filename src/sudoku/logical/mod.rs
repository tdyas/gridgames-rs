//! Logical Sudoku solving strategies.
//!
//! This module contains various logical solving strategies for Sudoku puzzles.

mod hidden_single;
mod single;

pub use hidden_single::HiddenSingleSolveStrategy;
pub use single::SinglePossibleSolveStrategy;
