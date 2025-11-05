//! Sudoku solver.
//!
//! This module provides a Sudoku solver built on top of the Dancing Links (`dlx`) implementation
//! in the [`crate::dlx`] module.
//!
//! The Dlx constaint matrix uses 324 columns arranged as 81 cell-occupancy constraints followed by
//! 27 zones × 9 digits each enforcing that every digit appears exactly once per zone.

use std::{borrow::Cow, collections::HashSet, fmt, num::NonZeroU8, str::FromStr};

use crate::dlx::{Dlx, SolveAction};

/// Metadata describing the constraint graph for a Sudoku-like puzzle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneMetadata {
    // Total number of cells in the game.
    pub num_cells: usize,

    // Number of values allowed in the same (e.g, for Sudoku 1-9 are the values).
    pub num_values: usize,

    // Define each zone by the cell indices for that zone.
    pub zones: Vec<Vec<usize>>,

    /// For each cell, the list of other cells that share at least one zone.
    pub neighbors_for_cell: Vec<Vec<usize>>,

    /// For each cell, the zones the cell is a member of.
    pub zones_for_cell: Vec<Vec<usize>>,
}

impl ZoneMetadata {
    /// Builds derived neighbor and zone lookup tables for the provided zone definitions.
    pub fn new(num_cells: usize, num_values: usize, zones: Vec<Vec<usize>>) -> ZoneMetadata {
        let mut neighbor_sets: Vec<HashSet<usize>> = vec![HashSet::new(); num_cells];
        let mut zones_for_cell_sets: Vec<HashSet<usize>> = vec![HashSet::new(); num_cells];

        for (zone_index, zone_cells) in zones.iter().enumerate() {
            for &cell in zone_cells {
                assert!(
                    cell < num_cells,
                    "zone index {zone_index} references cell {cell} outside 0..{num_cells}"
                );
                zones_for_cell_sets[cell].insert(zone_index);
            }

            for &cell in zone_cells {
                for &other in zone_cells {
                    if cell != other {
                        neighbor_sets[cell].insert(other);
                    }
                }
            }
        }

        let neighbors_for_cell = neighbor_sets
            .into_iter()
            .map(|set| {
                let mut neighbors: Vec<usize> = set.into_iter().collect();
                neighbors.sort_unstable();
                neighbors
            })
            .collect();

        let zones_for_cell = zones_for_cell_sets
            .into_iter()
            .map(|set| {
                let mut zone_list: Vec<usize> = set.into_iter().collect();
                zone_list.sort_unstable();
                zone_list
            })
            .collect();

        ZoneMetadata {
            num_cells,
            num_values,
            zones,
            neighbors_for_cell,
            zones_for_cell,
        }
    }
}

/// Classic 9x9 Sudoku constraint graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SudokuGraph {
    pub metadata: ZoneMetadata,
}

impl SudokuGraph {
    pub const NUM_CELLS: usize = 9 * 9;
    pub const NUM_VALUES: usize = 9;
    pub const NUM_ZONES: usize = 27;

    /// Returns metadata for a standard 9x9 Sudoku puzzle (rows, columns, 3x3 boxes).
    pub fn new() -> SudokuGraph {
        let mut zones: Vec<Vec<usize>> = Vec::with_capacity(Self::NUM_ZONES);

        // Row zones.
        for row in 0..9 {
            let mut zone = Vec::with_capacity(9);
            for column in 0..9 {
                zone.push(row * 9 + column);
            }
            zones.push(zone);
        }

        // Column zones.
        for column in 0..9 {
            let mut zone = Vec::with_capacity(9);
            for row in 0..9 {
                zone.push(row * 9 + column);
            }
            zones.push(zone);
        }

        // 3x3 box zones.
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut zone = Vec::with_capacity(9);
                let row_origin = box_row * 3;
                let col_origin = box_col * 3;
                for row in 0..3 {
                    for col in 0..3 {
                        zone.push((row_origin + row) * 9 + (col_origin + col));
                    }
                }
                zones.push(zone);
            }
        }

        let metadata = ZoneMetadata::new(Self::NUM_CELLS, Self::NUM_VALUES, zones);
        SudokuGraph { metadata }
    }
}

/// Mutable Sudoku board that stores digits in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SudokuBoard {
    cells: [Option<NonZeroU8>; SudokuGraph::NUM_CELLS],
}

impl SudokuBoard {
    /// Creates an empty board (all cells unset).
    pub fn empty() -> SudokuBoard {
        SudokuBoard {
            cells: [None; SudokuGraph::NUM_CELLS],
        }
    }

    /// Returns the stored digit at the provided cell index (`None` for empty).
    pub fn value(&self, index: usize) -> Option<u8> {
        assert!(index < SudokuGraph::NUM_CELLS);
        self.cells[index].map(|digit| digit.get())
    }

    /// Sets a digit (1-9) or clears a cell by providing `None`.
    pub fn set_value(&mut self, index: usize, value: Option<u8>) -> Result<(), String> {
        if index >= SudokuGraph::NUM_CELLS {
            return Err(format!("cell index {index} is out of bounds"));
        }
        match value {
            Some(digit @ 1..=9) => {
                self.cells[index] = NonZeroU8::new(digit);
                Ok(())
            }
            Some(digit) => Err(format!("value {digit} is outside the allowed range 1-9")),
            None => {
                self.cells[index] = None;
                Ok(())
            }
        }
    }

    /// Removes any digit from the given cell.
    pub fn clear(&mut self, index: usize) {
        assert!(index < SudokuGraph::NUM_CELLS);
        self.cells[index] = None;
    }

    /// Iterator over indices that currently have givens.
    pub fn given_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| value.map(|_| idx))
    }

    /// Returns an owning string representation using '.' for empty cells.
    pub fn to_puzzle_string(&self) -> String {
        self.cells
            .iter()
            .map(|value| match value {
                None => '.',
                Some(digit) => (b'0' + digit.get()) as char,
            })
            .collect()
    }
}

impl Default for SudokuBoard {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for SudokuBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..9 {
            if row != 0 {
                writeln!(f)?;
            }
            for col in 0..9 {
                match self.value(row * 9 + col) {
                    Some(value) => write!(f, "{value}")?,
                    None => write!(f, ".")?,
                }
                if col != 8 {
                    write!(f, " ")?
                }
            }
        }
        Ok(())
    }
}

impl FromStr for SudokuBoard {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = normalize_puzzle_string(s);

        if trimmed.len() != SudokuGraph::NUM_CELLS {
            return Err(format!(
                "expected {} characters, found {}",
                SudokuGraph::NUM_CELLS,
                trimmed.len()
            ));
        }

        let mut board = SudokuBoard::empty();
        for (idx, ch) in trimmed.chars().enumerate() {
            let value = match ch {
                '1'..='9' => NonZeroU8::new(ch as u8 - b'0'),
                '0' | '.' => None,
                _ => {
                    return Err(format!(
                        "invalid character '{ch}' at position {idx}; expected digits or '.'"
                    ))
                }
            };
            board.cells[idx] = value;
        }

        Ok(board)
    }
}

fn normalize_puzzle_string(s: &str) -> Cow<'_, str> {
    if s.len() == SudokuGraph::NUM_CELLS {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s.chars().filter(|c| !c.is_whitespace()).collect())
    }
}

/// Sudoku solver that leverages the DLX exact-cover implementation.
///
/// The underlying exact-cover matrix uses the column ordering described in the module docs:
/// one column per cell, followed by zone×digit columns. Each candidate `(cell, digit)` row
/// asserts the cell column and the three zone columns corresponding to the cell's row, column,
/// and box. Givens are applied by pre-selecting the associated rows before invoking `solve`.
pub struct SudokuDlxSolver {
    graph: SudokuGraph,
    dlx: Dlx,
    row_assignments: Vec<(usize, u8)>,
}

impl SudokuDlxSolver {
    pub fn new() -> SudokuDlxSolver {
        SudokuDlxSolver::with_graph(SudokuGraph::new())
    }

    pub fn with_graph(graph: SudokuGraph) -> SudokuDlxSolver {
        let num_cell_columns = graph.metadata.num_cells;
        let num_zone_value_columns = graph.metadata.zones.len() * graph.metadata.num_values;
        let num_columns = num_cell_columns + num_zone_value_columns;

        let mut dlx = Dlx::new(num_columns);
        let mut row_assignments =
            Vec::with_capacity(graph.metadata.num_cells * graph.metadata.num_values);

        let mut row_buffer = vec![false; num_columns];
        for cell_index in 0..graph.metadata.num_cells {
            for value in 1..=graph.metadata.num_values {
                row_buffer.fill(false);

                // Cell constraint: each cell must contain exactly one digit.
                row_buffer[cell_index] = true;

                // Zone/value constraints: each zone contains each digit at most once.
                for &zone_index in &graph.metadata.zones_for_cell[cell_index] {
                    let column =
                        num_cell_columns + zone_index * graph.metadata.num_values + (value - 1);
                    row_buffer[column] = true;
                }

                dlx.push_row(&row_buffer);
                row_assignments.push((cell_index, value as u8));
            }
        }

        SudokuDlxSolver {
            graph,
            dlx,
            row_assignments,
        }
    }

    /// Computes a solved board for the provided puzzle, returning up to two solutions.
    pub fn solve(&mut self, board: &SudokuBoard) -> Vec<SudokuBoard> {
        self.solve_with_limit(board, Some(2))
    }

    /// Computes Sudoku solutions, optionally limiting how many solutions are produced.
    pub fn solve_with_limit(
        &mut self,
        board: &SudokuBoard,
        limit: Option<usize>,
    ) -> Vec<SudokuBoard> {
        let max_solutions = limit.unwrap_or(usize::MAX);
        let mut solutions = Vec::new();

        self.dlx.clear_solution();

        for cell_index in board.given_indices() {
            if let Some(value) = board.value(cell_index) {
                let row_index = Self::row_index_for_assignment(
                    cell_index,
                    value,
                    self.graph.metadata.num_values,
                );
                self.dlx.add_row_to_solution(row_index);
            }
        }

        self.dlx.solve(|solution_rows| {
            let mut solved = board.clone();
            for row_index in solution_rows {
                let (cell_index, digit) = self.row_assignments[row_index];
                solved
                    .set_value(cell_index, Some(digit))
                    .expect("solver should only emit valid digits");
            }
            solutions.push(solved);

            if solutions.len() >= max_solutions {
                SolveAction::Stop
            } else {
                SolveAction::Continue
            }
        });

        self.dlx.clear_solution();
        solutions
    }

    fn row_index_for_assignment(cell_index: usize, value: u8, num_values: usize) -> usize {
        assert!(value >= 1 && value as usize <= num_values);
        cell_index * num_values + (value as usize - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_metadata_has_expected_geometry() {
        let graph = SudokuGraph::new();
        let metadata = graph.metadata;

        assert_eq!(metadata.num_cells, 81);
        assert_eq!(metadata.num_values, 9);
        assert_eq!(metadata.zones.len(), 27);

        for zone in metadata.zones.iter() {
            assert_eq!(zone.len(), 9);
        }

        // Each Sudoku cell should have 20 unique neighbors.
        for neighbors in metadata.neighbors_for_cell.iter() {
            assert_eq!(neighbors.len(), 20);
        }

        // Cell (row 0, col 0) participates in row 0, column 9, and box 18.
        assert_eq!(metadata.zones_for_cell[0], vec![0, 9, 18]);
        // Cell (row 4, col 7) participates in row 4, column 16, and box 23.
        assert_eq!(metadata.zones_for_cell[4 * 9 + 7], vec![4, 16, 23]);
    }

    #[test]
    fn board_from_str_and_round_trip() {
        let puzzle = "\
            530070000\
            600195000\
            098000060\
            800060003\
            400803001\
            700020006\
            060000280\
            000419005\
            000080079";
        let board: SudokuBoard = puzzle.parse().expect("valid puzzle");
        assert_eq!(board.value(0), Some(5));
        assert_eq!(board.value(1), Some(3));
        assert_eq!(board.value(2), None);
        assert_eq!(board.value(80), Some(9));

        let reconstructed = board.to_puzzle_string();
        let expected: String = puzzle
            .chars()
            .map(|c| if c == '0' { '.' } else { c })
            .collect();
        assert_eq!(reconstructed, expected);
    }

    #[test]
    fn board_set_value_validates_input() {
        let mut board = SudokuBoard::empty();
        assert!(board.set_value(10, Some(5)).is_ok());
        assert_eq!(board.value(10), Some(5));

        assert!(board.set_value(10, None).is_ok());
        assert_eq!(board.value(10), None);

        let err = board.set_value(100, Some(1)).unwrap_err();
        assert!(err.contains("out of bounds"));

        let err = board.set_value(0, Some(12)).unwrap_err();
        assert!(err.contains("outside the allowed range"));
    }

    #[test]
    fn dlx_solver_solves_classic_puzzle() {
        let mut solver = SudokuDlxSolver::new();
        let puzzle = "\
            530070000\
            600195000\
            098000060\
            800060003\
            400803001\
            700020006\
            060000280\
            000419005\
            000080079";
        let board: SudokuBoard = puzzle.parse().expect("valid puzzle");
        let solutions = solver.solve(&board);

        assert_eq!(solutions.len(), 1);
        let solved = &solutions[0];
        let solution_string = solved.to_puzzle_string();
        assert!(!solution_string.contains('.'));
        assert_eq!(
            solution_string,
            normalize_puzzle_string(
                "\
                534678912\
                672195348\
                198342567\
                859761423\
                426853791\
                713924856\
                961537284\
                287419635\
                345286179"
            )
        );
    }

    #[test]
    fn dlx_solver_finds_multiple_solutions_for_sparse_puzzle() {
        let mut solver = SudokuDlxSolver::new();
        let puzzle = "\
            100000000\
            020000000\
            003000000\
            000400000\
            000050000\
            000006000\
            000000700\
            000000080\
            000000009";
        let board: SudokuBoard = puzzle.parse().expect("valid puzzle");
        let solutions = solver.solve_with_limit(&board, Some(2));

        assert_eq!(
            solutions.len(),
            2,
            "expected solver to stop after two solutions"
        );
        for solved in solutions {
            assert!(!solved.to_puzzle_string().contains('.'));
        }
    }

    #[test]
    fn dlx_solver_returns_zero_for_contradictory_board() {
        let mut solver = SudokuDlxSolver::new();
        let puzzle = "\
            550070000\
            600195000\
            098000060\
            800060003\
            400803001\
            700020006\
            060000280\
            000419005\
            000080079";
        let board: SudokuBoard = puzzle.parse().expect("valid puzzle format");
        let solutions = solver.solve(&board);
        assert!(solutions.is_empty());
    }

    #[test]
    fn dlx_solver_is_reusable_between_runs() {
        let mut solver = SudokuDlxSolver::new();
        let puzzle = "\
            530070000\
            600195000\
            098000060\
            800060003\
            400803001\
            700020006\
            060000280\
            000419005\
            000080079";
        let board: SudokuBoard = puzzle.parse().expect("valid puzzle");

        let first = solver.solve(&board);
        let second = solver.solve(&board);

        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
    }

    #[test]
    fn solution_rows_cover_all_columns() {
        let mut solver = SudokuDlxSolver::new();
        let puzzle = "\
            530070000\
            600195000\
            098000060\
            800060003\
            400803001\
            700020006\
            060000280\
            000419005\
            000080079";
        let board: SudokuBoard = puzzle.parse().expect("valid puzzle");

        solver.dlx.clear_solution();
        for cell_index in board.given_indices() {
            if let Some(value) = board.value(cell_index) {
                let row_index = SudokuDlxSolver::row_index_for_assignment(
                    cell_index,
                    value,
                    solver.graph.metadata.num_values,
                );
                solver.dlx.add_row_to_solution(row_index);
            }
        }

        let mut captured_rows = Vec::new();
        solver.dlx.solve(|rows| {
            captured_rows = rows;
            SolveAction::Stop
        });

        solver.dlx.clear_solution();

        let num_columns = solver.graph.metadata.num_cells
            + solver.graph.metadata.zones.len() * solver.graph.metadata.num_values;
        assert_eq!(captured_rows.len(), solver.graph.metadata.num_cells);

        let mut column_counts = vec![0u8; num_columns];
        for row_index in captured_rows {
            let (cell_index, digit) = solver.row_assignments[row_index];

            if let Some(given_digit) = board.value(cell_index) {
                assert_eq!(given_digit, digit);
            }

            column_counts[cell_index] += 1;
            for &zone_index in &solver.graph.metadata.zones_for_cell[cell_index] {
                let column = solver.graph.metadata.num_cells
                    + zone_index * solver.graph.metadata.num_values
                    + (digit as usize - 1);
                column_counts[column] += 1;
            }
        }

        assert!(column_counts.iter().all(|&count| count == 1));
    }
}
