//! gridgames-rs
//!
//! This crate contains various Sudoku solvers and generators. It is intended to support
//! the `GridGames` iOS app.

#![deny(warnings)]
#![allow(dead_code)]

pub mod board;
pub mod dlx;
pub mod sudoku;

// Re-export main types for convenience
pub use board::{Board, FindResult, SolveStrategy, SolverMove};
pub use sudoku::{SudokuBoard, ZoneMetadata};
