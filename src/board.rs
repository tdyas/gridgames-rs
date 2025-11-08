//! Generic board for grid-based logic puzzles.
//!
//! This module provides the [`Board`] struct which represents any grid-based constraint
//! satisfaction puzzle (e.g., Sudoku, Latin Squares, etc.).

use std::collections::HashMap;
use std::num::NonZeroU8;
use std::sync::Arc;

use crate::sudoku::{SudokuBoard, ZoneMetadata};

/// A constraint-propagation board for solving grid-based logic puzzles.
/// Tracks cell values, possible values, and statistics during solving.
#[derive(Clone, Debug)]
pub struct Board {
    /// Reference to the constraint graph defining game rules
    metadata: Arc<ZoneMetadata>,

    /// Cell values: None = empty, Some(v) = filled with value v (1-N)
    values: Vec<Option<NonZeroU8>>,

    /// Bitmask of possible values per cell (bit i = value i+1 is possible)
    possible: Vec<u32>,

    /// Count of unfilled cells per zone
    zone_counts: Vec<usize>,

    /// Number of filled cells
    num_set_cells: usize,

    /// Statistics tracking which solving techniques were used and how often
    /// Key examples: "singles", "hidden_singles", "backtrack", etc.
    stats: HashMap<String, usize>,

    /// History of moves made during solving
    /// Used for step-by-step solution playback and UI feedback
    moves: Vec<SolverMove>,
}

/// Represents a single solving move/step
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverMove {
    /// Cell index where the move was made
    pub index: usize,

    /// Value that was set (1-N, never 0)
    pub value: NonZeroU8,

    /// Technique that found this move (e.g., "single", "hidden_single")
    pub technique: String,
}

/// Result from searching for the cell with the least possibilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FindResult {
    /// Found a cell with the given index and possibility count
    Found(usize),
    /// Board is completely solved (all cells filled)
    Solved,
    /// Board has a contradiction (unsolvable state)
    Contradiction,
}


impl Board {
    /// Creates a new empty board with the given zone graph.
    pub fn new(metadata: Arc<ZoneMetadata>) -> Self {
        let num_cells = metadata.num_cells;
        let num_zones = metadata.zones.len();
        let all_values_mask = (1 << metadata.num_values) - 1;

        // Initialize zone counts - each zone starts with all cells unfilled
        let mut zone_counts = Vec::with_capacity(num_zones);
        for zone in &metadata.zones {
            zone_counts.push(zone.len());
        }

        Board {
            metadata,
            values: vec![None; num_cells],
            possible: vec![all_values_mask; num_cells],
            zone_counts,
            num_set_cells: 0,
            stats: HashMap::new(),
            moves: Vec::new(),
        }
    }

    /// Creates a board from initial cell values.
    /// Values should be None for empty, Some(v) for filled cells with value v (1-N)
    pub fn from_values(metadata: Arc<ZoneMetadata>, values: &[Option<u8>]) -> Result<Self, String> {
        if values.len() != metadata.num_cells {
            return Err(format!(
                "Expected {} values, got {}",
                metadata.num_cells,
                values.len()
            ));
        }

        let mut board = Self::new(metadata);
        for (index, &value) in values.iter().enumerate() {
            if let Some(v) = value {
                board.set_value(index, v)?;
            }
        }
        Ok(board)
    }

    /// Sets a cell to a specific value, propagating constraints to neighbors.
    /// Returns an error if the value is invalid or creates a contradiction.
    pub fn set_value(&mut self, index: usize, value: u8) -> Result<(), String> {
        // Validate inputs
        if index >= self.metadata.num_cells {
            return Err(format!("index {} out of bounds", index));
        }
        if value == 0 || value > self.metadata.num_values as u8 {
            return Err(format!(
                "value {} out of range 1..={}",
                value, self.metadata.num_values
            ));
        }

        // Check if value is possible
        let value_bit = 1 << (value - 1);
        if self.possible[index] & value_bit == 0 {
            return Err(format!("value {} not possible at index {}", value, index));
        }

        // Convert to NonZeroU8 (safe because we validated value >= 1)
        let non_zero_value = NonZeroU8::new(value).unwrap();

        // Update cell value
        if self.values[index].is_none() {
            self.num_set_cells += 1;

            // Update zone counts
            for &zone_index in &self.metadata.zones_for_cell[index] {
                self.zone_counts[zone_index] -= 1;
            }
        }

        self.values[index] = Some(non_zero_value);
        self.possible[index] = 0; // No other values possible

        // Update all neighbors - remove this value from their possibility sets
        for &neighbor_index in &self.metadata.neighbors_for_cell[index] {
            self.possible[neighbor_index] &= !value_bit;
        }

        Ok(())
    }

    /// Clears a cell value and recalculates constraints
    pub fn reset_value(&mut self, index: usize) -> Result<(), String> {
        if index >= self.metadata.num_cells {
            return Err(format!("index {} out of bounds", index));
        }

        if self.values[index].is_some() {
            self.values[index] = None;
            self.num_set_cells -= 1;

            // Update zone counts
            for &zone_index in &self.metadata.zones_for_cell[index] {
                self.zone_counts[zone_index] += 1;
            }

            // Recalculate possible values for this cell
            self.recalculate_possible(index);

            // Recalculate possible values for all neighbors
            // Clone the neighbor list to avoid borrow checker issues
            let neighbors: Vec<usize> = self.metadata.neighbors_for_cell[index].clone();
            for neighbor_index in neighbors {
                self.recalculate_possible(neighbor_index);
            }
        }

        Ok(())
    }

    /// Recalculates the possible values for a cell based on its neighbors
    fn recalculate_possible(&mut self, index: usize) {
        if self.values[index].is_some() {
            self.possible[index] = 0;
            return;
        }

        // Start with all values possible
        let mut mask = (1 << self.metadata.num_values) - 1;

        // Remove values used by neighbors
        for &neighbor_index in &self.metadata.neighbors_for_cell[index] {
            if let Some(neighbor_value) = self.values[neighbor_index] {
                mask &= !(1 << (neighbor_value.get() - 1));
            }
        }

        self.possible[index] = mask;
    }

    // ===== Querying State =====

    /// Gets the value at a cell (None = empty, Some(v) = filled with value v)
    pub fn get_value(&self, index: usize) -> Option<u8> {
        self.values.get(index).and_then(|v| v.map(|nz| nz.get()))
    }

    /// Checks if a cell is empty
    pub fn is_empty(&self, index: usize) -> bool {
        self.values.get(index).map(|v| v.is_none()).unwrap_or(true)
    }

    /// Gets all cell values as a flat array (None = empty, Some(v) = filled with value v)
    pub fn get_all_values(&self) -> Vec<Option<u8>> {
        self.values.iter().map(|v| v.map(|nz| nz.get())).collect()
    }

    /// Gets the bitmask of possible values for a cell
    /// Bit i represents whether value i+1 is possible
    pub fn get_possible(&self, index: usize) -> u32 {
        self.possible.get(index).copied().unwrap_or(0)
    }

    /// Gets the count of possible values for a cell
    pub fn count_possible(&self, index: usize) -> u32 {
        self.get_possible(index).count_ones()
    }

    /// Checks if a specific value is possible at a cell
    pub fn is_value_possible(&self, index: usize, value: u8) -> bool {
        if value == 0 || value > self.metadata.num_values as u8 {
            return false;
        }
        let value_bit = 1 << (value - 1);
        self.get_possible(index) & value_bit != 0
    }

    /// Gets the list of possible values for a cell as a Vec
    pub fn get_possible_values(&self, index: usize) -> Vec<u8> {
        let mask = self.get_possible(index);
        (0..self.metadata.num_values)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| (i + 1) as u8)
            .collect()
    }

    // ===== Solving Support =====

    /// Finds the empty cell with the fewest possible values (best branching heuristic)
    /// Returns:
    /// - `FindResult::Solved` if the board is completely solved
    /// - `FindResult::Contradiction` if the board has an unsolvable contradiction
    /// - `FindResult::Found(index)` with the cell index having the fewest possibilities
    pub fn find_index_with_least_possibilities(&self) -> FindResult {
        if self.num_set_cells == self.metadata.num_cells {
            return FindResult::Solved;
        }

        let mut least_index: Option<usize> = None;
        let mut least_count = u32::MAX;

        for (index, &poss) in self.possible.iter().enumerate() {
            if self.values[index].is_none() {
                let count = poss.count_ones();
                if count == 0 {
                    return FindResult::Contradiction; // Contradiction found
                }
                if count < least_count {
                    least_count = count;
                    least_index = Some(index);
                }
            }
        }

        least_index.map(FindResult::Found).unwrap_or(FindResult::Contradiction)
    }

    /// Checks if the board is solved (all cells filled with no contradictions)
    pub fn is_solved(&self) -> bool {
        matches!(self.find_index_with_least_possibilities(), FindResult::Solved)
    }

    /// Checks if the board has a contradiction (unsolvable state)
    pub fn has_contradiction(&self) -> bool {
        self.values
            .iter()
            .enumerate()
            .any(|(idx, val)| val.is_none() && self.possible[idx] == 0)
    }

    // ===== Statistics & Moves =====

    /// Increments a statistic counter (e.g., technique usage)
    pub fn inc_stat(&mut self, key: &str) {
        *self.stats.entry(key.to_string()).or_insert(0) += 1;
    }

    /// Gets the value of a statistic counter
    pub fn get_stat(&self, key: &str) -> usize {
        self.stats.get(key).copied().unwrap_or(0)
    }

    /// Gets all statistics
    pub fn get_stats(&self) -> &HashMap<String, usize> {
        &self.stats
    }

    /// Clears all statistics
    pub fn clear_stats(&mut self) {
        self.stats.clear();
    }

    /// Records a move in the history
    pub fn record_move(&mut self, index: usize, value: u8, technique: impl Into<String>) {
        // value should always be 1-N, never 0, so this should always succeed
        if let Some(nz_value) = NonZeroU8::new(value) {
            self.moves.push(SolverMove {
                index,
                value: nz_value,
                technique: technique.into(),
            });
        }
    }

    /// Gets all recorded moves
    pub fn get_moves(&self) -> &[SolverMove] {
        &self.moves
    }

    /// Clears the move history
    pub fn clear_moves(&mut self) {
        self.moves.clear();
    }

    // ===== Metadata Access =====

    /// Gets the number of cells in the board
    pub fn num_cells(&self) -> usize {
        self.metadata.num_cells
    }

    /// Gets the number of possible values (1-N)
    pub fn num_values(&self) -> usize {
        self.metadata.num_values
    }

    /// Gets the number of filled cells
    pub fn num_set_cells(&self) -> usize {
        self.num_set_cells
    }

    /// Gets the constraint metadata (graph)
    pub fn metadata(&self) -> &Arc<ZoneMetadata> {
        &self.metadata
    }

    /// Gets the count of unfilled cells in a specific zone
    pub fn zone_count(&self, zone_index: usize) -> usize {
        self.zone_counts.get(zone_index).copied().unwrap_or(0)
    }

    // ===== SudokuBoard Integration =====

    /// Creates a Board from a SudokuBoard (9x9 specific)
    pub fn from_sudoku_board(
        board: &SudokuBoard,
        metadata: Arc<ZoneMetadata>,
    ) -> Result<Board, String> {
        if metadata.num_cells != 81 {
            return Err("SudokuBoard requires 81-cell metadata".to_string());
        }

        let mut b = Board::new(metadata);
        for index in 0..81 {
            if let Some(value) = board.value(index) {
                b.set_value(index, value)?;
            }
        }
        Ok(b)
    }

    /// Converts to a SudokuBoard (9x9 specific)
    pub fn to_sudoku_board(&self) -> Result<SudokuBoard, String> {
        if self.metadata.num_cells != 81 {
            return Err("SudokuBoard requires 81 cells".to_string());
        }

        let mut board = SudokuBoard::empty();
        for (index, &value) in self.values.iter().enumerate() {
            board.set_value(index, value.map(|v| v.get()))?;
        }
        Ok(board)
    }

    // ===== Debug Support =====

    /// Prints debug information about the board state
    /// Similar to C++ Board::Dump()
    pub fn dump(&self, message: &str) {
        println!("=== {} ===", message);
        println!(
            "Cells set: {}/{}",
            self.num_set_cells, self.metadata.num_cells
        );

        // Print values grid (assuming square grid for formatting)
        let side = (self.metadata.num_cells as f64).sqrt() as usize;
        println!("\nValues:");
        for row in 0..side {
            for col in 0..side {
                let idx = row * side + col;
                match self.get_value(idx) {
                    Some(val) => print!(" {} ", val),
                    None => print!(" . "),
                }
            }
            println!();
        }

        // Print possibility counts
        println!("\nPossibility counts:");
        for row in 0..side {
            for col in 0..side {
                let idx = row * side + col;
                let count = self.count_possible(idx);
                if self.is_empty(idx) {
                    print!(" {} ", count);
                } else {
                    print!(" - ");
                }
            }
            println!();
        }

        // Print statistics
        if !self.stats.is_empty() {
            println!("\nStatistics:");
            let mut stats: Vec<_> = self.stats.iter().collect();
            stats.sort_by_key(|(k, _)| *k);
            for (key, value) in stats {
                println!("  {}: {}", key, value);
            }
        }

        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sudoku::SudokuGraph;

    fn make_sudoku_metadata() -> Arc<ZoneMetadata> {
        Arc::new(SudokuGraph::new().metadata)
    }

    #[test]
    fn test_new_board() {
        let metadata = make_sudoku_metadata();
        let board = Board::new(metadata.clone());

        assert_eq!(board.num_cells(), 81);
        assert_eq!(board.num_values(), 9);
        assert_eq!(board.num_set_cells(), 0);

        // All cells should be empty
        for i in 0..81 {
            assert_eq!(board.get_value(i), None);
            assert!(board.is_empty(i));
        }

        // All cells should have all 9 values possible
        for i in 0..81 {
            assert_eq!(board.count_possible(i), 9);
            let expected_mask = (1 << 9) - 1; // 0b111111111
            assert_eq!(board.get_possible(i), expected_mask);
        }
    }

    #[test]
    fn test_set_value_basic() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Set cell 0 (row 0, col 0) to value 5.
        assert!(board.set_value(0, 5).is_ok());
        assert_eq!(board.get_value(0), Some(5));
        assert_eq!(board.num_set_cells(), 1);
        assert_eq!(board.count_possible(0), 0); // No other values possible

        // Value 5 should be removed from all neighbors.
        // Check same row (cells 1-8)
        for col in 1..9 {
            assert!(!board.is_value_possible(col, 5));
        }

        // Check same column (cells 9, 18, 27, ...).
        for row in 1..9 {
            assert!(!board.is_value_possible(row * 9, 5));
        }

        // Check same box (cells 1, 2, 9, 10, 11, 18, 19, 20)
        for &idx in &[1, 2, 9, 10, 11, 18, 19, 20] {
            assert!(!board.is_value_possible(idx, 5));
        }
    }

    #[test]
    fn test_set_value_constraint_propagation() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Fill the first row with values 1-9
        for col in 0..9 {
            board.set_value(col, (col + 1) as u8).unwrap();
        }

        // All cells in row 0 should be filled
        assert_eq!(board.num_set_cells(), 9);

        // Cell 9 (row 1, col 0) should not be able to have value 1
        // because cell 0 (row 0, col 0) already has it
        assert!(!board.is_value_possible(9, 1));

        // Cell 1 (row 0, col 1) should have no possibilities
        assert_eq!(board.count_possible(1), 0);
    }

    #[test]
    fn test_set_value_invalid_index() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        let result = board.set_value(100, 5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of bounds"));
    }

    #[test]
    fn test_set_value_invalid_value() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Value 0 is invalid
        let result = board.set_value(0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));

        // Value 10 is invalid for Sudoku (max is 9)
        let result = board.set_value(0, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));
    }

    #[test]
    fn test_set_value_impossible() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Set cell 0 to value 5
        board.set_value(0, 5).unwrap();

        // Try to set cell 1 (same row) to value 5 - should fail
        let result = board.set_value(1, 5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not possible"));
    }

    #[test]
    fn test_reset_value() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Set cell 0 to value 5
        board.set_value(0, 5).unwrap();
        assert_eq!(board.get_value(0), Some(5));
        assert_eq!(board.num_set_cells(), 1);

        // Reset cell 0
        board.reset_value(0).unwrap();
        assert_eq!(board.get_value(0), None);
        assert_eq!(board.num_set_cells(), 0);
        assert!(board.is_empty(0));

        // Cell 0 should have all values possible again
        assert_eq!(board.count_possible(0), 9);

        // Cell 1 should have value 5 possible again
        assert!(board.is_value_possible(1, 5));
    }

    #[test]
    fn test_from_values() {
        let metadata = make_sudoku_metadata();

        // Create a simple pattern
        let mut values = vec![None; 81];
        values[0] = Some(5);
        values[1] = Some(3);
        values[9] = Some(7);

        let board = Board::from_values(metadata.clone(), &values).unwrap();

        assert_eq!(board.get_value(0), Some(5));
        assert_eq!(board.get_value(1), Some(3));
        assert_eq!(board.get_value(9), Some(7));
        assert_eq!(board.num_set_cells(), 3);

        // Check constraint propagation happened
        assert!(!board.is_value_possible(2, 5)); // Same row as cell 0
        assert!(!board.is_value_possible(2, 3)); // Same row as cell 1
    }

    #[test]
    fn test_from_values_wrong_length() {
        let metadata = make_sudoku_metadata();
        let values = vec![None; 50]; // Wrong length

        let result = Board::from_values(metadata, &values);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 81"));
    }

    #[test]
    fn test_get_possible_values() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Initially, all values 1-9 should be possible
        let possible = board.get_possible_values(0);
        assert_eq!(possible, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Set cell 1 (same row) to value 5
        board.set_value(1, 5).unwrap();

        // Cell 0 should not have 5 as a possibility
        let possible = board.get_possible_values(0);
        assert_eq!(possible, vec![1, 2, 3, 4, 6, 7, 8, 9]);
    }

    #[test]
    fn test_find_index_with_least_possibilities() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Empty board - all cells have 9 possibilities
        let result = board.find_index_with_least_possibilities();
        assert!(matches!(result, FindResult::Found(_))); // Should return some cell

        // Fill row 0 except cell 0
        for col in 1..9 {
            board.set_value(col, col as u8).unwrap();
        }

        // Cell 0 should now have fewer possibilities
        // It can't be 1, 2, 3, 4, 5, 6, 7, or 8 (same row)
        let count = board.count_possible(0);
        assert_eq!(count, 1); // Only value 9 is possible

        let result = board.find_index_with_least_possibilities();
        assert_eq!(result, FindResult::Found(0)); // Cell 0 has the fewest possibilities
    }

    #[test]
    fn test_find_index_solved() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Fill entire board with a valid Sudoku solution
        let solution = vec![
            5, 3, 4, 6, 7, 8, 9, 1, 2, 6, 7, 2, 1, 9, 5, 3, 4, 8, 1, 9, 8, 3, 4, 2, 5, 6, 7, 8, 5,
            9, 7, 6, 1, 4, 2, 3, 4, 2, 6, 8, 5, 3, 7, 9, 1, 7, 1, 3, 9, 2, 4, 8, 5, 6, 9, 6, 1, 5,
            3, 7, 2, 8, 4, 2, 8, 7, 4, 1, 9, 6, 3, 5, 3, 4, 5, 2, 8, 6, 1, 7, 9,
        ];

        for (idx, &val) in solution.iter().enumerate() {
            board.set_value(idx, val).unwrap();
        }

        let result = board.find_index_with_least_possibilities();
        assert_eq!(result, FindResult::Solved);
        assert!(board.is_solved());
    }

    #[test]
    fn test_find_index_contradiction() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Create a contradiction by filling row 0 with 1-8
        for col in 0..8 {
            board.set_value(col, (col + 1) as u8).unwrap();
        }

        // Now fill column 8 with values 1-8 (different from row values)
        // This will make cell 8 (row 0, col 8) have no possibilities
        board.set_value(17, 9).unwrap(); // Row 1, col 8

        // Fill the rest of column 8 to eliminate possibilities for cell 8
        for row in 2..9 {
            let idx = row * 9 + 8;
            // Skip if value already used in row 0
            let val = if row == 2 { 1 } else { (row - 1) as u8 };
            if board.is_value_possible(idx, val) {
                board.set_value(idx, val).unwrap();
            }
        }

        // Check if we created a contradiction
        if board.has_contradiction() {
            let result = board.find_index_with_least_possibilities();
            assert_eq!(result, FindResult::Contradiction);
        }
    }

    #[test]
    fn test_stats() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        assert_eq!(board.get_stat("singles"), 0);

        board.inc_stat("singles");
        assert_eq!(board.get_stat("singles"), 1);

        board.inc_stat("singles");
        board.inc_stat("hidden_singles");
        assert_eq!(board.get_stat("singles"), 2);
        assert_eq!(board.get_stat("hidden_singles"), 1);

        board.clear_stats();
        assert_eq!(board.get_stat("singles"), 0);
    }

    #[test]
    fn test_moves() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        assert_eq!(board.get_moves().len(), 0);

        board.record_move(0, 5, "single");
        board.record_move(1, 3, "hidden_single");

        let moves = board.get_moves();
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0].index, 0);
        assert_eq!(moves[0].value.get(), 5);
        assert_eq!(moves[0].technique, "single");
        assert_eq!(moves[1].index, 1);
        assert_eq!(moves[1].value.get(), 3);
        assert_eq!(moves[1].technique, "hidden_single");

        board.clear_moves();
        assert_eq!(board.get_moves().len(), 0);
    }

    #[test]
    fn test_clone() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        board.set_value(0, 5).unwrap();
        board.inc_stat("test");

        let board2 = board.clone();

        assert_eq!(board2.get_value(0), Some(5));
        assert_eq!(board2.get_stat("test"), 1);
        assert_eq!(board2.num_set_cells(), 1);
    }

    #[test]
    fn test_sudoku_board_conversion() {
        let metadata = make_sudoku_metadata();

        // Create a SudokuBoard with some values
        let mut sudoku = SudokuBoard::empty();
        sudoku.set_value(0, Some(5)).unwrap();
        sudoku.set_value(1, Some(3)).unwrap();
        sudoku.set_value(10, Some(7)).unwrap();

        // Convert to Board
        let board = Board::from_sudoku_board(&sudoku, metadata.clone()).unwrap();
        assert_eq!(board.get_value(0), Some(5));
        assert_eq!(board.get_value(1), Some(3));
        assert_eq!(board.get_value(10), Some(7));
        assert_eq!(board.num_set_cells(), 3);

        // Convert back to SudokuBoard
        let sudoku2 = board.to_sudoku_board().unwrap();
        assert_eq!(sudoku2.value(0), Some(5));
        assert_eq!(sudoku2.value(1), Some(3));
        assert_eq!(sudoku2.value(10), Some(7));
        assert_eq!(sudoku2.value(2), None);
    }

    #[test]
    fn test_zone_counts() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata.clone());

        // Initially all zones should have 9 unfilled cells
        // Sudoku has 27 zones (9 rows + 9 columns + 9 boxes)
        for zone_idx in 0..27 {
            assert_eq!(board.zone_count(zone_idx), 9);
        }

        // Set cell 0 (row 0, col 0, box 0) to value 5
        board.set_value(0, 5).unwrap();

        // Zone 0 (row 0), zone 9 (col 0), and zone 18 (box 0) should each have 8 unfilled cells
        assert_eq!(board.zone_count(0), 8); // Row 0
        assert_eq!(board.zone_count(9), 8); // Column 0
        assert_eq!(board.zone_count(18), 8); // Box 0

        // Other zones should still have 9
        assert_eq!(board.zone_count(1), 9); // Row 1
    }
}
