//! Generic board for grid-based logic puzzles.
//!
//! This module provides the [`Board`] struct which represents any grid-based constraint
//! satisfaction puzzle (e.g., Sudoku, Latin Squares, etc.).

use std::collections::HashMap;
use std::num::NonZeroU8;
use std::sync::Arc;

use crate::gamedef::{GameDefinition, GameDefinitionError};

/// Compute the next set of solver moves for a particular [`Board`]. This is
/// implemented by each strategy.
pub trait SolveStrategy<GD: GameDefinition + Default, const CAP: usize> {
    /// Return a set of possible moves given the represented strategy.
    fn compute_solver_moves(board: &Board<GD, CAP>) -> Vec<SolverMove>;
}

/// A constraint-propagation board for solving grid-based logic puzzles.
/// Tracks cell values, possible values, and statistics during solving.
#[derive(Clone, Debug)]
pub struct Board<GD: GameDefinition + Default, const CAP: usize> {
    /// Reference to the constraint graph defining game rules
    gamedef: Arc<GD>,

    /// Cell values: None = empty, Some(v) = filled with value v (1-N).
    values: [Option<NonZeroU8>; CAP],

    /// Bitmask of possible values per cell (bit i = value i+1 is possible)
    possible: [u32; CAP],

    /// Count of each value in every zone (flattened: zone_index * num_values + value_index)
    zone_value_counts: Vec<u16>,

    /// Number of conflicting values (counts >= 2) per zone
    zone_conflict_counts: Vec<u16>,

    /// Total zones currently containing a conflict
    conflicted_zone_total: usize,

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

impl<GD: GameDefinition + Default, const CAP: usize> Board<GD, CAP> {
    /// Creates a new empty board with the given zone graph.
    pub fn new() -> Self {
        let gamedef = GD::default();
        let num_zones = gamedef.num_zones();
        let all_values_mask = (1 << gamedef.num_values()) - 1;
        let num_values = gamedef.num_values() as usize;

        // Initialize zone counts - each zone starts with all cells unfilled
        let mut zone_counts = Vec::with_capacity(num_zones);
        for zone_index in 0..num_zones {
            let zone = gamedef
                .get_cells_for_zone(zone_index)
                .expect("invalid zone index?");
            zone_counts.push(zone.len());
        }

        let zone_value_counts = vec![0u16; num_zones * num_values];
        let zone_conflict_counts = vec![0u16; num_zones];

        Board {
            gamedef: Arc::new(gamedef),
            values: [None; CAP],
            possible: [all_values_mask; CAP],
            zone_value_counts,
            zone_conflict_counts,
            conflicted_zone_total: 0,
            zone_counts,
            num_set_cells: 0,
            stats: HashMap::new(),
            moves: Vec::new(),
        }
    }

    /// Creates a board from initial cell values.
    /// Values should be None for empty, Some(v) for filled cells with value v (1-N)
    ///
    /// # Panics
    /// Panics if the length of `values` does not match `metadata.num_cells`.
    /// Conflicting assignments are allowed and tracked as zone conflicts.
    pub fn from_cell_values(values: &[Option<u8>]) -> Result<Self, String> {
        let mut board = Self::new();
        assert_eq!(
            values.len(),
            board.num_cells(),
            "Expected {} values, got {}",
            board.num_cells(),
            values.len()
        );

        for (index, &value) in values.iter().enumerate() {
            if let Some(v) = value {
                board.set_cell(index, v);
            }
        }
        Ok(board)
    }

    /// Creates a board from a puzzle string.
    /// Conflicting assignments are allowed and recorded for contradiction detection.
    pub fn from_puzzle_str(values_str: &str) -> Result<Self, String> {
        let mut board = Self::new();
        assert_eq!(
            values_str.len(),
            board.num_cells(),
            "Expected {} length string with values, got {} length instead",
            board.num_cells(),
            values_str.len()
        );
        for (index, ch) in values_str.chars().enumerate() {
            if ch.is_ascii_digit() {
                let value = (ch as u8) - b'0';
                if value > 0 {
                    board.set_cell(index, value);
                }
            } else if ch != '.' && !ch.is_whitespace() {
                return Err(format!("Expected a digit or '.', instead got `{ch}`"));
            }
        }
        Ok(board)
    }

    /// Sets a cell to a specific value, propagating constraints to neighbors.
    /// This operation is infallible; invalid indices or values will panic.
    /// Conflicting assignments are permitted and recorded for contradiction detection.
    pub fn set_cell(&mut self, index: usize, value: u8) {
        assert!(
            index < self.gamedef.num_cells(),
            "index {} out of bounds",
            index
        );
        assert!(
            value >= 1 && value <= self.gamedef.num_values(),
            "value {} out of range 1..={}",
            value,
            self.gamedef.num_values()
        );

        // If the cell already has this value, nothing to do.
        if let Some(existing) = self.values[index] {
            if existing.get() == value {
                return;
            }
            // Replace an existing value by clearing first to keep counts consistent.
            self.clear_cell(index)
                .expect("clearing cell should succeed");
        }

        let value_bit = 1 << (value - 1);
        let non_zero_value = NonZeroU8::new(value).unwrap();

        // Update cell value
        if self.values[index].is_none() {
            self.num_set_cells += 1;

            // Update zone counts and conflict tracking
            for &zone_index in self
                .gamedef
                .get_zones_for_cell(index)
                .expect("set_cell failed to get zones for cell")
            {
                self.zone_counts[zone_index] -= 1;

                let offset = self.zone_value_offset(zone_index, value);
                let previous = self.zone_value_counts[offset];
                self.zone_value_counts[offset] = previous + 1;
                if previous == 1 {
                    if self.zone_conflict_counts[zone_index] == 0 {
                        self.conflicted_zone_total += 1;
                    }
                    self.zone_conflict_counts[zone_index] += 1;
                }
            }
        }

        self.values[index] = Some(non_zero_value);
        self.possible[index] = 0; // No other values possible

        // Update all neighbors - remove this value from their possibility sets
        for &neighbor_index in self.gamedef.get_neighbors_for_cell(index).unwrap() {
            self.possible[neighbor_index] &= !value_bit;
        }
    }

    /// Clears a cell value and recalculates constraints
    pub fn clear_cell(&mut self, index: usize) -> Result<(), String> {
        if index >= self.gamedef.num_cells() {
            return Err(format!("index {} out of bounds", index));
        }

        if let Some(prev_value) = self.values[index] {
            self.values[index] = None;
            self.num_set_cells -= 1;

            // Update the zone's count of unfilled cells.
            for &zone_index in self.gamedef.get_zones_for_cell(index).unwrap() {
                self.zone_counts[zone_index] += 1;

                let offset = self.zone_value_offset(zone_index, prev_value.get());
                let previous = self.zone_value_counts[offset];
                debug_assert!(previous > 0, "clearing cell with zero count recorded");
                self.zone_value_counts[offset] = previous - 1;

                if previous == 2 {
                    self.zone_conflict_counts[zone_index] -= 1;
                    if self.zone_conflict_counts[zone_index] == 0 {
                        self.conflicted_zone_total -= 1;
                    }
                }
            }

            // Recalculate the possible values for this cell.
            self.recalculate_possible(index);

            // Recalculate possible values for all neighboring cells as well.
            // Clone the Arc to avoid holding an immutable borrow of `self` while mutating.
            let gamedef = Arc::clone(&self.gamedef);
            for &neighbor_index in gamedef.get_neighbors_for_cell(index).unwrap() {
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

        // Start with all values possible.
        let mut mask = (1 << self.gamedef.num_values()) - 1;

        // Remove values used by neighbors.
        for &neighbor_index in self.gamedef.get_neighbors_for_cell(index).unwrap() {
            if let Some(neighbor_value) = self.values[neighbor_index] {
                mask &= !(1 << (neighbor_value.get() - 1));
            }
        }

        self.possible[index] = mask;
    }

    #[inline]
    fn zone_value_offset(&self, zone_index: usize, value: u8) -> usize {
        zone_index * self.gamedef.num_values() as usize + (value as usize - 1)
    }

    /// Gets the value at a cell (None = empty, Some(v) = filled with value v)
    pub fn get_cell(&self, index: usize) -> Option<u8> {
        self.values.get(index).and_then(|v| v.map(|nz| nz.get()))
    }

    /// Checks if a cell is empty
    pub fn is_cell_empty(&self, index: usize) -> bool {
        self.values.get(index).map(|v| v.is_none()).unwrap_or(true)
    }

    /// Gets all cell values as a flat array (None = empty, Some(v) = filled with value v)
    pub fn get_all_cell_values(&self) -> Vec<Option<u8>> {
        self.values.iter().map(|v| v.map(|nz| nz.get())).collect()
    }

    /// Gets the bitmask of possible values for a cell
    /// Bit i represents whether value i+1 is possible
    pub fn get_possible_valus_mask_for_cell(&self, index: usize) -> u32 {
        self.possible.get(index).copied().unwrap_or(0)
    }

    /// Gets the count of possible values for a cell
    pub fn count_possible_values_for_cell(&self, index: usize) -> usize {
        self.get_possible_valus_mask_for_cell(index).count_ones() as usize
    }

    /// Checks if a specific value is possible at a cell
    pub fn is_value_possible(&self, index: usize, value: u8) -> bool {
        if value == 0 || value > self.gamedef.num_values() {
            return false;
        }
        let value_bit = 1 << (value - 1);
        self.get_possible_valus_mask_for_cell(index) & value_bit != 0
    }

    /// Gets the list of possible values for a cell as a Vec
    pub fn get_possible_values_for_cell(&self, index: usize) -> Vec<u8> {
        let mask = self.get_possible_valus_mask_for_cell(index);
        (0..self.gamedef.num_values())
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| i + 1)
            .collect()
    }

    pub fn given_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.values
            .iter()
            .take(self.gamedef.num_cells())
            .enumerate()
            .filter_map(|(index, value)| value.map(|_| index))
    }

    pub fn to_puzzle_string(&self) -> String {
        self.values
            .iter()
            .take(self.gamedef.num_cells())
            .map(|value| match value {
                None => '.',
                Some(digit) => (b'0' + digit.get()) as char,
            })
            .collect()
    }

    // ===== Solving Support =====

    /// Finds the empty cell with the fewest possible values (best branching heuristic)
    /// Returns:
    /// - `FindResult::Solved` if the board is completely solved
    /// - `FindResult::Contradiction` if the board has an unsolvable contradiction
    /// - `FindResult::Found(index)` with the cell index having the fewest possibilities
    pub fn find_index_with_least_possibilities(&self) -> FindResult {
        if self.has_zone_conflict() {
            return FindResult::Contradiction;
        }

        if self.num_set_cells == self.gamedef.num_cells() {
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

        least_index
            .map(FindResult::Found)
            .unwrap_or(FindResult::Contradiction)
    }

    /// Checks if the board is solved (all cells filled with no contradictions)
    pub fn is_solved(&self) -> bool {
        matches!(
            self.find_index_with_least_possibilities(),
            FindResult::Solved
        )
    }

    /// Checks if the board has a contradiction (unsolvable state)
    pub fn has_contradiction(&self) -> bool {
        if self.has_zone_conflict() {
            return true;
        }

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

    /// Gets the count of unfilled cells in a specific zone
    pub fn zone_count(&self, zone_index: usize) -> usize {
        self.zone_counts.get(zone_index).copied().unwrap_or(0)
    }

    /// Returns true if any zone currently contains duplicate values.
    pub fn has_zone_conflict(&self) -> bool {
        self.conflicted_zone_total > 0
    }

    /// Returns how many zones currently contain at least one conflicting value.
    pub fn num_conflicted_zones(&self) -> usize {
        self.conflicted_zone_total
    }

    /// Returns the number of conflicting value entries in a zone (values set twice or more).
    pub fn zone_conflict_count(&self, zone_index: usize) -> u16 {
        self.zone_conflict_counts
            .get(zone_index)
            .copied()
            .unwrap_or(0)
    }

    // ===== Debug Support =====

    /// Prints debug information about the board state
    /// Similar to C++ Board::Dump()
    pub fn dump(&self, message: &str) {
        println!("=== {} ===", message);
        println!(
            "Cells set: {}/{}",
            self.num_set_cells,
            self.gamedef.num_cells()
        );

        // Print values grid (assuming square grid for formatting)
        let side = (self.gamedef.num_cells() as f64).sqrt() as usize;
        println!("\nValues:");
        for row in 0..side {
            for col in 0..side {
                let idx = row * side + col;
                match self.get_cell(idx) {
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
                let count = self.count_possible_values_for_cell(idx);
                if self.is_cell_empty(idx) {
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

impl<GD: GameDefinition + Default, const CAP: usize> GameDefinition for Board<GD, CAP> {
    #[inline]
    fn num_cells(&self) -> usize {
        self.gamedef.num_cells()
    }

    #[inline]
    fn num_values(&self) -> u8 {
        self.gamedef.num_values()
    }

    #[inline]
    fn num_zones(&self) -> usize {
        self.gamedef.num_zones()
    }

    #[inline]
    fn get_cells_for_zone(&self, zone_index: usize) -> Result<&[usize], GameDefinitionError> {
        self.gamedef.get_cells_for_zone(zone_index)
    }

    #[inline]
    fn get_neighbors_for_cell(&self, cell_index: usize) -> Result<&[usize], GameDefinitionError> {
        self.gamedef.get_neighbors_for_cell(cell_index)
    }

    #[inline]
    fn get_zones_for_cell(&self, cell_index: usize) -> Result<&[usize], GameDefinitionError> {
        self.gamedef.get_zones_for_cell(cell_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sudoku::SudokuBoard;

    #[test]
    fn test_new_board() {
        let board = SudokuBoard::new();

        assert_eq!(board.num_cells(), 81);
        assert_eq!(board.num_values(), 9);
        assert_eq!(board.given_indices().count(), 0);

        // All cells should be empty
        for i in 0..81 {
            assert_eq!(board.get_cell(i), None);
            assert!(board.is_cell_empty(i));
        }

        // All cells should have all 9 values possible
        for i in 0..81 {
            assert_eq!(board.count_possible_values_for_cell(i), 9);
            let expected_mask = (1 << 9) - 1; // 0b111111111
            assert_eq!(board.get_possible_valus_mask_for_cell(i), expected_mask);
        }
    }

    #[test]
    fn test_set_value_basic() {
        let mut board = SudokuBoard::new();

        // Set cell 0 (row 0, col 0) to value 5.
        board.set_cell(0, 5);
        assert_eq!(board.get_cell(0), Some(5));
        assert_eq!(board.given_indices().count(), 1);
        assert_eq!(board.count_possible_values_for_cell(0), 0); // No other values possible

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
        let mut board = SudokuBoard::new();

        // Fill the first row with values 1-9
        for col in 0..9 {
            board.set_cell(col, (col + 1) as u8);
        }

        // All cells in row 0 should be filled
        assert_eq!(board.given_indices().count(), 9);

        // Cell 9 (row 1, col 0) should not be able to have value 1
        // because cell 0 (row 0, col 0) already has it
        assert!(!board.is_value_possible(9, 1));

        // Cell 1 (row 0, col 1) should have no possibilities
        assert_eq!(board.count_possible_values_for_cell(1), 0);
    }

    #[test]
    fn test_set_value_invalid_index() {
        let mut board = SudokuBoard::new();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            board.set_cell(100, 5);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_set_value_invalid_value() {
        let mut board = SudokuBoard::new();

        // Value 0 is invalid
        let zero_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            board.set_cell(0, 0);
        }));
        assert!(zero_result.is_err());

        // Value 10 is invalid for Sudoku (max is 9)
        let high_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            board.set_cell(0, 10);
        }));
        assert!(high_result.is_err());
    }

    #[test]
    fn test_set_value_impossible() {
        let mut board = SudokuBoard::new();

        // Set cell 0 to value 5
        board.set_cell(0, 5);

        // Try to set cell 1 (same row) to value 5 - now allowed but records conflict
        board.set_cell(1, 5);
        assert!(board.has_zone_conflict());
        assert!(
            board.zone_conflict_count(0) >= 1,
            "Row 0 should report a conflict"
        );
        assert!(
            board.zone_conflict_count(18) >= 1,
            "Box 0 should report a conflict"
        );
    }

    #[test]
    fn test_reset_value() {
        let mut board = SudokuBoard::new();

        // Set cell 0 to value 5
        board.set_cell(0, 5);
        assert_eq!(board.get_cell(0), Some(5));
        assert_eq!(board.given_indices().count(), 1);

        // Reset cell 0
        board.clear_cell(0).unwrap();
        assert_eq!(board.get_cell(0), None);
        assert_eq!(board.given_indices().count(), 0);
        assert!(board.is_cell_empty(0));

        // Cell 0 should have all values possible again
        assert_eq!(board.count_possible_values_for_cell(0), 9);

        // Cell 1 should have value 5 possible again
        assert!(board.is_value_possible(1, 5));
    }

    #[test]
    fn test_from_values() {
        // Create a simple pattern
        let mut values = vec![None; 81];
        values[0] = Some(5);
        values[1] = Some(3);
        values[9] = Some(7);

        let board = SudokuBoard::from_cell_values(&values).unwrap();

        assert_eq!(board.get_cell(0), Some(5));
        assert_eq!(board.get_cell(1), Some(3));
        assert_eq!(board.get_cell(9), Some(7));
        assert_eq!(board.given_indices().count(), 3);

        // Check constraint propagation happened
        assert!(!board.is_value_possible(2, 5)); // Same row as cell 0
        assert!(!board.is_value_possible(2, 3)); // Same row as cell 1
    }

    #[test]
    #[should_panic(expected = "Expected 81 values, got 50")]
    fn test_from_values_wrong_length() {
        let values = vec![None; 50]; // Wrong length
        let _ = SudokuBoard::from_cell_values(&values).unwrap();
    }

    #[test]
    fn test_get_possible_values() {
        let mut board = SudokuBoard::new();

        // Initially, all values 1-9 should be possible
        let possible = board.get_possible_values_for_cell(0);
        assert_eq!(possible, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Set cell 1 (same row) to value 5
        board.set_cell(1, 5);

        // Cell 0 should not have 5 as a possibility
        let possible = board.get_possible_values_for_cell(0);
        assert_eq!(possible, vec![1, 2, 3, 4, 6, 7, 8, 9]);
    }

    #[test]
    fn test_find_index_with_least_possibilities() {
        let mut board = SudokuBoard::new();

        // Empty board - all cells have 9 possibilities
        let result = board.find_index_with_least_possibilities();
        assert!(matches!(result, FindResult::Found(_))); // Should return some cell

        // Fill row 0 except cell 0
        for col in 1..9 {
            board.set_cell(col, col as u8);
        }

        // Cell 0 should now have fewer possibilities
        // It can't be 1, 2, 3, 4, 5, 6, 7, or 8 (same row)
        let count = board.count_possible_values_for_cell(0);
        assert_eq!(count, 1); // Only value 9 is possible

        let result = board.find_index_with_least_possibilities();
        assert_eq!(result, FindResult::Found(0)); // Cell 0 has the fewest possibilities
    }

    #[test]
    fn test_find_index_solved() {
        let mut board = SudokuBoard::new();

        // Fill entire board with a valid Sudoku solution
        let solution = vec![
            5, 3, 4, 6, 7, 8, 9, 1, 2, 6, 7, 2, 1, 9, 5, 3, 4, 8, 1, 9, 8, 3, 4, 2, 5, 6, 7, 8, 5,
            9, 7, 6, 1, 4, 2, 3, 4, 2, 6, 8, 5, 3, 7, 9, 1, 7, 1, 3, 9, 2, 4, 8, 5, 6, 9, 6, 1, 5,
            3, 7, 2, 8, 4, 2, 8, 7, 4, 1, 9, 6, 3, 5, 3, 4, 5, 2, 8, 6, 1, 7, 9,
        ];

        for (idx, &val) in solution.iter().enumerate() {
            board.set_cell(idx, val);
        }

        let result = board.find_index_with_least_possibilities();
        assert_eq!(result, FindResult::Solved);
        assert!(board.is_solved());
    }

    #[test]
    fn test_find_index_contradiction() {
        let mut board = SudokuBoard::new();

        // Create a contradiction by filling row 0 with 1-8
        for col in 0..8 {
            board.set_cell(col, (col + 1) as u8);
        }

        // Now fill column 8 with values 1-8 (different from row values)
        // This will make cell 8 (row 0, col 8) have no possibilities
        board.set_cell(17, 9); // Row 1, col 8

        // Fill the rest of column 8 to eliminate possibilities for cell 8
        for row in 2..9 {
            let idx = row * 9 + 8;
            // Skip if value already used in row 0
            let val = if row == 2 { 1 } else { (row - 1) as u8 };
            if board.is_value_possible(idx, val) {
                board.set_cell(idx, val);
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
        let mut board = SudokuBoard::new();

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
        let mut board = SudokuBoard::new();

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
        let mut board = SudokuBoard::new();

        board.set_cell(0, 5);
        board.inc_stat("test");

        let board2 = board.clone();

        assert_eq!(board2.get_cell(0), Some(5));
        assert_eq!(board2.get_stat("test"), 1);
        assert_eq!(board2.given_indices().count(), 1);
    }

    #[test]
    fn test_zone_counts() {
        let mut board = SudokuBoard::new();

        // Initially all zones should have 9 unfilled cells
        // Sudoku has 27 zones (9 rows + 9 columns + 9 boxes)
        for zone_idx in 0..27 {
            assert_eq!(board.zone_count(zone_idx), 9);
        }

        // Set cell 0 (row 0, col 0, box 0) to value 5
        board.set_cell(0, 5);

        // Zone 0 (row 0), zone 9 (col 0), and zone 18 (box 0) should each have 8 unfilled cells
        assert_eq!(board.zone_count(0), 8); // Row 0
        assert_eq!(board.zone_count(9), 8); // Column 0
        assert_eq!(board.zone_count(18), 8); // Box 0

        // Other zones should still have 9
        assert_eq!(board.zone_count(1), 9); // Row 1
    }
}
