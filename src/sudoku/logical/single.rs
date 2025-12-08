//! Single (naked single) solving strategy.

use crate::{
    board::{Board, SolveStrategy, SolverMove},
    gamedef::GameDefinition,
};
use std::num::NonZeroU8;

/// Strategy that finds cells with only one possible value (naked singles).
pub struct SinglePossibleSolveStrategy;

impl<GD: GameDefinition + Default, const CAP: usize> SolveStrategy<GD, CAP>
    for SinglePossibleSolveStrategy
{
    /// Finds all cells that have exactly one possible value.
    fn compute_solver_moves(board: &Board<GD, CAP>) -> Vec<SolverMove> {
        let mut moves = Vec::new();

        for index in 0..board.num_cells() {
            // Skip cells that are already filled
            if board.get_cell(index).is_some() {
                continue;
            }

            let possible_values = board.get_possible_values_for_cell(index);
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
    use std::collections::BTreeSet;

    use super::*;
    use crate::sudoku::SudokuBoard;

    #[test]
    fn test_single_possible_empty_board() {
        // Empty board has no singles (all cells have 9 possibilities).
        let board = SudokuBoard::new();
        let moves = SinglePossibleSolveStrategy::compute_solver_moves(&board);
        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_single_possible_finds_naked_single() {
        let mut board = SudokuBoard::new();

        // Fill row 0 except cell 0, leaving only one possibility (9) for cell 0.
        for col in 1..9 {
            board.set_cell(col, col as u8).unwrap();
        }

        let moves = SinglePossibleSolveStrategy::compute_solver_moves(&board);

        // Should find cell 0 with value 9.
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].index, 0);
        assert_eq!(moves[0].value.get(), 9);
        assert_eq!(moves[0].technique, "single");
    }

    #[test]
    fn test_single_possible_multiple_singles() {
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

        let board = SudokuBoard::from_puzzle_str(puzzle_str).unwrap();

        let moves = SinglePossibleSolveStrategy::compute_solver_moves(&board);

        // This puzzle should have at least one naked single
        assert!(
            !moves.is_empty(),
            "Puzzle should have at least one naked single"
        );

        let actual: BTreeSet<(usize, u8)> =
            moves.iter().map(|m| (m.index, m.value.get())).collect();
        let expected: BTreeSet<(usize, u8)> = BTreeSet::from([(59, 7), (62, 4), (70, 3)]);
        assert_eq!(
            actual, expected,
            "Should find the expected naked singles for this puzzle"
        );

        // Verify all moves are valid naked singles
        for mov in &moves {
            assert_eq!(mov.technique, "single");
            assert!(
                board.is_value_possible(mov.index, mov.value.get()),
                "Move at cell {} with value {} should be possible on the board",
                mov.index,
                mov.value.get()
            );
            let possible = board.get_possible_values_for_cell(mov.index);
            assert_eq!(
                possible.len(),
                1,
                "Cell {} should only have one possible value, but has: {:?}",
                mov.index,
                possible
            );
            assert_eq!(
                possible[0],
                mov.value.get(),
                "The only possible value should match the move"
            );
        }
    }
}
