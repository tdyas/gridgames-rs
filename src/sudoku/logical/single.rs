//! Single (naked single) solving strategy.

use crate::board::{Board, SolveStrategy, SolverMove};
use std::num::NonZeroU8;

/// Strategy that finds cells with only one possible value (naked singles).
pub struct SinglePossibleSolveStrategy;

impl SolveStrategy for SinglePossibleSolveStrategy {
    /// Finds all cells that have exactly one possible value.
    fn compute_solver_moves(board: &Board) -> Vec<SolverMove> {
        let mut moves = Vec::new();

        for index in 0..board.num_cells() {
            // Skip cells that are already filled
            if board.get_value(index).is_some() {
                continue;
            }

            let possible_values = board.get_possible_values(index);
            if possible_values.len() == 1 {
                let value = possible_values[0];
                if let Some(nz_value) = NonZeroU8::new(value) {
                    moves.push(SolverMove {
                        index,
                        value: nz_value,
                        technique: "single".to_string(),
                    });
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
    fn test_single_possible_empty_board() {
        let metadata = make_sudoku_metadata();
        let board = Board::new(metadata);

        // Empty board has no singles (all cells have 9 possibilities)
        let moves = SinglePossibleSolveStrategy::compute_solver_moves(&board);
        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_single_possible_finds_naked_single() {
        let metadata = make_sudoku_metadata();
        let mut board = Board::new(metadata);

        // Fill row 0 except cell 0, leaving only one possibility (9) for cell 0
        for col in 1..9 {
            board.set_value(col, col as u8).unwrap();
        }

        let moves = SinglePossibleSolveStrategy::compute_solver_moves(&board);

        // Should find cell 0 with value 9
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].index, 0);
        assert_eq!(moves[0].value.get(), 9);
        assert_eq!(moves[0].technique, "single");
    }

    #[test]
    fn test_single_possible_multiple_singles() {
        let metadata = make_sudoku_metadata();

        // Create a puzzle with multiple naked singles
        let puzzle_str = "\
            53.......\
            6..195...\
            .98....6.\
            8...6...3\
            4..8.3..1\
            7...2...6\
            .6....28.\
            ...419..5\
            ....8..79";

        let sudoku: SudokuBoard = puzzle_str.parse().unwrap();
        let board = Board::from_sudoku_board(&sudoku, metadata).unwrap();

        let moves = SinglePossibleSolveStrategy::compute_solver_moves(&board);

        // There should be at least one single in this position
        assert!(!moves.is_empty());
        for mov in &moves {
            assert_eq!(mov.technique, "single");
        }
    }
}
