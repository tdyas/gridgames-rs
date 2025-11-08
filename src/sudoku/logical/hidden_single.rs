//! Hidden single solving strategy.

use crate::board::{Board, SolveStrategy, SolverMove};
use std::num::NonZeroU8;

/// Strategy that finds values that can only go in one cell within a zone (hidden singles).
pub struct HiddenSingleSolveStrategy;

impl SolveStrategy for HiddenSingleSolveStrategy {
    /// Finds all values that can only go in one cell within each zone.
    fn compute_solver_moves(board: &Board) -> Vec<SolverMove> {
        let mut moves = Vec::new();
        let metadata = board.metadata();

        // For each zone, check each value
        for zone in &metadata.zones {
            for value in 1..=metadata.num_values as u8 {
                let mut possible_cells = Vec::new();

                // Find all cells in this zone where this value is possible.
                for &cell_index in zone {
                    // Skip if cell is already filled
                    if board.get_value(cell_index).is_some() {
                        continue;
                    }

                    if board.is_value_possible(cell_index, value) {
                        possible_cells.push(cell_index);
                    }
                }

                // If exactly one cell can hold this value, it's a hidden single
                if possible_cells.len() == 1 {
                    let index = possible_cells[0];

                    // Only record if this cell has multiple possibilities
                    // (if it has only one possibility, it's a naked single, not hidden).
                    if board.count_possible(index) > 1 {
                        if let Some(nz_value) = NonZeroU8::new(value) {
                            moves.push(SolverMove {
                                index,
                                value: nz_value,
                                technique: "hidden_single".to_string(),
                            });
                        }
                    }
                }
            }
        }

        moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sudoku::{SudokuBoard, SudokuGraph};

    fn make_sudoku_metadata() -> crate::sudoku::ZoneMetadata {
        SudokuGraph::new().metadata
    }

    #[test]
    fn test_hidden_single_empty_board() {
        let metadata = make_sudoku_metadata();
        let board = Board::new(metadata);

        // Empty board has no hidden singles
        let moves = HiddenSingleSolveStrategy::compute_solver_moves(&board);
        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_hidden_single_in_row() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata);

        // Set up a situation where value 9 can only go in one cell in row 0
        // Fill cells in row 0 with values 1-8, leaving cells 0 and 8 empty
        for col in 1..8 {
            board.set_value(col, col as u8).unwrap();
        }

        // Now eliminate 9 from cell 0 by placing 9 in the same column
        board.set_value(9, 9).unwrap(); // Cell (1, 0) - same column as cell 0

        // Now in row 0, only cell 8 can have value 9 (hidden single)
        let moves = HiddenSingleSolveStrategy::compute_solver_moves(&board);

        // Should find at least the hidden single for 9 in cell 8
        let move_9_in_cell_8 = moves.iter().find(|m| m.index == 8 && m.value.get() == 9);
        assert!(
            move_9_in_cell_8.is_some(),
            "Should find hidden single: value 9 at cell 8"
        );
        assert_eq!(move_9_in_cell_8.unwrap().technique, "hidden_single");
    }

    #[test]
    fn test_hidden_single_in_box() {
        // Create a clearer scenario:
        // We want 9 to only be possible in cell 1 within box 0
        // Fill most of box 0, leaving only cells 0 and 1
        // Then eliminate 9 from cell 0

        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata);

        // Fill box 0 (cells: 0, 1, 2, 9, 10, 11, 18, 19, 20) except cells 0 and 1
        board.set_value(2, 1).unwrap();
        board.set_value(9, 2).unwrap();
        board.set_value(10, 3).unwrap();
        board.set_value(11, 4).unwrap();
        board.set_value(18, 5).unwrap();
        board.set_value(19, 6).unwrap();
        board.set_value(20, 7).unwrap();

        // Eliminate 9 from cell 0 by placing it in the same column
        board.set_value(27, 9).unwrap(); // Cell (3, 0) - same column as cell 0

        // Now 9 can only go in cell 1 within box 0 (hidden single)
        let moves = HiddenSingleSolveStrategy::compute_solver_moves(&board);

        let move_9_in_cell_1 = moves.iter().find(|m| m.index == 1 && m.value.get() == 9);
        assert!(
            move_9_in_cell_1.is_some(),
            "Should find hidden single: value 9 at cell 1 in box 0. Found moves: {:?}",
            moves.iter().filter(|m| m.technique == "hidden_single").collect::<Vec<_>>()
        );
        assert_eq!(move_9_in_cell_1.unwrap().technique, "hidden_single");
    }

    #[test]
    fn test_hidden_single_vs_naked_single() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata);

        // Create a naked single (cell with only one possibility)
        // Fill row 0 except cell 0
        for col in 1..9 {
            board.set_value(col, col as u8).unwrap();
        }

        // Cell 0 now has only one possibility (9), which is a naked single
        let hidden_moves = HiddenSingleSolveStrategy::compute_solver_moves(&board);

        // HiddenSingleSolveStrategy should NOT report it as a hidden single
        // (it filters out cells with only one possibility)
        let cell_0_in_hidden = hidden_moves.iter().find(|m| m.index == 0);
        assert!(
            cell_0_in_hidden.is_none(),
            "Naked singles should not be reported as hidden singles"
        );
    }

    #[test]
    fn test_strategies_with_real_puzzle() {
        let metadata = make_sudoku_metadata();

        // A real Sudoku puzzle
        let puzzle_str = "\
            530070000\
            600195000\
            098000060\
            800060003\
            400803001\
            700020006\
            060000280\
            000419005\
            000080079";

        let sudoku: SudokuBoard = puzzle_str.parse().unwrap();
        let board = Board::from_sudoku_board(&sudoku, metadata).unwrap();

        let hidden_singles = HiddenSingleSolveStrategy::compute_solver_moves(&board);

        // Verify all moves are valid
        for mov in hidden_singles.iter() {
            assert!(
                mov.value.get() >= 1 && mov.value.get() <= 9,
                "Move value should be in range 1-9"
            );
            assert!(
                mov.index < 81,
                "Move index should be in range 0-80"
            );
            assert!(
                board.is_value_possible(mov.index, mov.value.get()),
                "Move should be possible on the board"
            );
        }
    }
}
