//! Sudoku solver.
//!
//! This module provides a Sudoku solver built on top of the Dancing Links (`dlx`) implementation
//! in the [`crate::dlx`] module.
//!
//! The Dlx constaint matrix uses 324 columns arranged as 81 cell-occupancy constraints followed by
//! 27 zones × 9 digits each enforcing that every digit appears exactly once per zone.

use std::sync::{Arc, OnceLock};

use crate::board::Board;
use crate::dlx::{Dlx, SolveAction};
use crate::gamedef::{GameDefinition, GameDefinitionError, GenericGameDefinition};

pub mod generate;
pub mod logical;

/// Type alias to make constructing Board instances for Sudoku easier. By using this type alias,
/// even with methods defined on `Board`, Rust will infer `CAP` properly when those methods are
/// invoked via this alias, e.g. `SudokuBoard::new`.
pub type SudokuBoard = Board<SudokuGameDefinition, 81>;

/// Sudoku game definition. The metadata contains the constraint graph for a Sudoku-like puzzle.
#[derive(Clone, Debug)]
/// Classic 9x9 Sudoku constraint graph.
pub struct SudokuGameDefinition {
    gamedef: Arc<GenericGameDefinition>,
}

impl SudokuGameDefinition {
    pub const NUM_CELLS: usize = 9 * 9;
    pub const NUM_VALUES: u8 = 9;
    pub const NUM_ZONES: usize = 27;

    /// Returns metadata for a standard 9x9 Sudoku puzzle (rows, columns, 3x3 boxes).
    pub fn new() -> Self {
        static GAMEDEF: OnceLock<Arc<GenericGameDefinition>> = OnceLock::new();
        let gamedef = GAMEDEF.get_or_init(|| {
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

            let gamedef = GenericGameDefinition::new(Self::NUM_CELLS, Self::NUM_VALUES, zones);
            Arc::new(gamedef)
        });

        Self {
            gamedef: Arc::clone(gamedef),
        }
    }
}

impl Default for SudokuGameDefinition {
    fn default() -> Self {
        Self::new()
    }
}

impl GameDefinition for SudokuGameDefinition {
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

/// Sudoku solver that leverages the DLX exact-cover implementation.
///
/// The underlying exact-cover matrix uses the column ordering described in the module docs:
/// one column per cell, followed by zone×digit columns. Each candidate `(cell, digit)` row
/// asserts the cell column and the three zone columns corresponding to the cell's row, column,
/// and box. Givens are applied by pre-selecting the associated rows before invoking `solve`.
pub struct SudokuDlxSolver {
    gamedef: SudokuGameDefinition,
    dlx: Dlx,
    row_assignments: Vec<(usize, u8)>,
}

impl SudokuDlxSolver {
    pub fn new() -> SudokuDlxSolver {
        let gamedef = SudokuGameDefinition::new();
        let num_cell_columns = gamedef.num_cells();
        let num_zone_value_columns = gamedef.num_zones() * gamedef.num_values() as usize;
        let num_columns = num_cell_columns + num_zone_value_columns;

        let mut dlx = Dlx::new(num_columns);
        let mut row_assignments =
            Vec::with_capacity(gamedef.num_cells() * gamedef.num_values() as usize);

        let mut row_buffer = vec![false; num_columns];
        for cell_index in 0..gamedef.num_cells() {
            for value in 1..=gamedef.num_values() {
                row_buffer.fill(false);

                // Cell constraint: each cell must contain exactly one digit.
                row_buffer[cell_index] = true;

                // Zone/value constraints: each zone contains each digit at most once.
                for &zone_index in gamedef.get_zones_for_cell(cell_index).unwrap() {
                    let column = num_cell_columns
                        + zone_index * gamedef.num_values() as usize
                        + (value as usize - 1);
                    row_buffer[column] = true;
                }

                dlx.push_row(&row_buffer);
                row_assignments.push((cell_index, value));
            }
        }

        SudokuDlxSolver {
            gamedef,
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
            if let Some(value) = board.get_value(cell_index) {
                let row_index = Self::row_index_for_assignment(
                    cell_index,
                    value,
                    self.gamedef.num_values() as usize,
                );
                self.dlx.add_row_to_solution(row_index);
            }
        }

        self.dlx.solve(|solution_rows| {
            let mut solved = board.clone();
            for row_index in solution_rows {
                let (cell_index, digit) = self.row_assignments[row_index];
                if solved.get_value(cell_index).is_none() {
                    solved
                        .set_value(cell_index, digit)
                        .expect("solver should only emit valid digits");
                }
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
        let gamedef = SudokuGameDefinition::new();

        assert_eq!(gamedef.num_cells(), 81);
        assert_eq!(gamedef.num_values(), 9);
        assert_eq!(gamedef.num_zones(), 27);

        for zone_index in 0..gamedef.num_zones() {
            let zone = gamedef.get_cells_for_zone(zone_index).unwrap();
            assert_eq!(zone.len(), 9);
        }

        // Each Sudoku cell should have 20 unique neighbors.
        for cell_index in 0..gamedef.num_cells() {
            let neighbors = gamedef.get_neighbors_for_cell(cell_index).unwrap();
            assert_eq!(neighbors.len(), 20);
        }

        // Cell (row 0, col 0) participates in row 0, column 9, and box 18.
        assert_eq!(gamedef.get_zones_for_cell(0).unwrap(), &[0, 9, 18]);
        // Cell (row 4, col 7) participates in row 4, column 16, and box 23.
        assert_eq!(gamedef.get_zones_for_cell(4 * 9 + 7).unwrap(), &[4, 16, 23]);
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
        let board = SudokuBoard::from_puzzle_str(puzzle).expect("valid puzzle");
        assert_eq!(board.get_value(0), Some(5));
        assert_eq!(board.get_value(1), Some(3));
        assert_eq!(board.get_value(2), None);
        assert_eq!(board.get_value(80), Some(9));

        let reconstructed = board.to_puzzle_string();
        let expected: String = puzzle
            .chars()
            .map(|c| if c == '0' { '.' } else { c })
            .collect();
        assert_eq!(reconstructed, expected);
    }

    #[test]
    fn board_set_value_validates_input() {
        let mut board = SudokuBoard::new();
        assert!(board.set_value(10, 5).is_ok());
        assert_eq!(board.get_value(10), Some(5));

        assert!(board.reset_value(10).is_ok());
        assert_eq!(board.get_value(10), None);

        let err = board.set_value(100, 1).unwrap_err();
        assert!(err.contains("out of bounds"));

        let err = board.set_value(0, 12).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn dlx_solver_solves_classic_puzzle() {
        let mut solver = SudokuDlxSolver::new();
        let puzzle = "\
            53..7....\
            6..195...\
            .98....6.\
            8...6...3\
            4..8.3..1\
            7...2...6\
            .6....28.\
            ...419..5\
            ....8..79";
        let board = SudokuBoard::from_puzzle_str(puzzle).expect("valid puzzle");
        let solutions = solver.solve(&board);

        assert_eq!(solutions.len(), 1);
        let solved = &solutions[0];
        let solution_string = solved.to_puzzle_string();
        assert!(!solution_string.contains('.'));
        assert_eq!(
            solution_string,
            ("\
                534678912\
                672195348\
                198342567\
                859761423\
                426853791\
                713924856\
                961537284\
                287419635\
                345286179")
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
        let board = SudokuBoard::from_puzzle_str(puzzle).expect("valid puzzle");
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
        let board = SudokuBoard::from_puzzle_str(puzzle).expect("valid puzzle");

        let first: Vec<_> = solver
            .solve(&board)
            .into_iter()
            .map(|b| b.to_puzzle_string())
            .collect();
        let second: Vec<_> = solver
            .solve(&board)
            .into_iter()
            .map(|b| b.to_puzzle_string())
            .collect();

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
        let board = SudokuBoard::from_puzzle_str(puzzle).expect("valid puzzle");

        solver.dlx.clear_solution();
        for cell_index in board.given_indices() {
            if let Some(value) = board.get_value(cell_index) {
                let row_index = SudokuDlxSolver::row_index_for_assignment(
                    cell_index,
                    value,
                    board.num_values().into(),
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

        let num_columns = board.num_cells() + board.num_zones() * board.num_values() as usize;
        assert_eq!(captured_rows.len(), board.num_cells());

        let mut column_counts = vec![0u8; num_columns];
        for row_index in captured_rows {
            let (cell_index, digit) = solver.row_assignments[row_index];

            if let Some(given_digit) = board.get_value(cell_index) {
                assert_eq!(given_digit, digit);
            }

            column_counts[cell_index] += 1;
            for &zone_index in board.get_zones_for_cell(cell_index).unwrap() {
                let column = board.num_cells()
                    + zone_index * board.num_values() as usize
                    + (digit as usize - 1);
                column_counts[column] += 1;
            }
        }

        assert!(column_counts.iter().all(|&count| count == 1));
    }
}
