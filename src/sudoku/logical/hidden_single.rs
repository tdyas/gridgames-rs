//! Hidden single solving strategy.

use std::collections::HashSet;
use std::num::NonZeroU8;

use crate::board::{Board, SolveStrategy, SolverMove};
use crate::gamedef::GameDefinition;

/// Strategy that finds values that can only go in one cell within a zone (hidden singles).
pub struct HiddenSingleSolveStrategy;

impl<GD: GameDefinition + Default, const CAPACITY: usize> SolveStrategy<GD, CAPACITY>
    for HiddenSingleSolveStrategy
{
    /// Finds all values that can only go in one cell within each zone.
    fn compute_solver_moves(board: &Board<GD, CAPACITY>) -> Vec<SolverMove> {
        let mut moves = Vec::new();
        let mut seen_moves = HashSet::new();

        // For each zone, check each value
        for zone_index in 0..board.num_zones() {
            let zone = board.get_cells_for_zone(zone_index).unwrap();
            for value in 1..=board.num_values() {
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
                    if board.count_possible(index) > 1
                        && let Some(nz_value) = NonZeroU8::new(value)
                        && seen_moves.insert((index, value))
                    {
                        moves.push(SolverMove {
                            index,
                            value: nz_value,
                            technique: "hidden_single".to_string(),
                        });
                    }
                }
            }
        }

        moves
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::sudoku::{SudokuBoard, SudokuGameDefinition};

    fn make_sudoku_board() -> SudokuBoard {
        SudokuBoard::new()
    }

    #[test]
    fn test_hidden_single_empty_board() {
        // An empty board has no hidden singles.
        let board = make_sudoku_board();
        let moves = HiddenSingleSolveStrategy::compute_solver_moves(&board);
        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_hidden_single_in_row() {
        let mut board = make_sudoku_board();

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

        let mut board = make_sudoku_board();

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
            moves
                .iter()
                .filter(|m| m.technique == "hidden_single")
                .collect::<Vec<_>>()
        );
        assert_eq!(move_9_in_cell_1.unwrap().technique, "hidden_single");
    }

    #[test]
    fn test_hidden_single_vs_naked_single() {
        let mut board = make_sudoku_board();

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

        let board: SudokuBoard = Board::from_str(SudokuGameDefinition::new(), puzzle_str).unwrap();

        let hidden_singles = HiddenSingleSolveStrategy::compute_solver_moves(&board);
        assert!(
            !hidden_singles.is_empty(),
            "Puzzle should produce at least one hidden single"
        );

        let unique_moves: HashSet<(usize, u8)> = hidden_singles
            .iter()
            .map(|m| (m.index, m.value.get()))
            .collect();

        assert_eq!(
            unique_moves.len(),
            hidden_singles.len(),
            "Hidden singles should be unique per cell/value pair: {:?}",
            hidden_singles
        );

        let expected_moves = [
            (5, 8),
            (7, 1),
            (22, 4),
            (24, 5),
            (38, 6),
            (42, 7),
            (47, 3),
            (51, 8),
            (54, 9),
            (64, 8),
            (69, 6),
            (78, 1),
        ];
        let expected_set: HashSet<(usize, u8)> = expected_moves.into_iter().collect();
        assert_eq!(
            unique_moves, expected_set,
            "Hidden singles should match expected moves for this puzzle"
        );

        // Verify all moves are valid hidden singles.
        for mov in hidden_singles.iter() {
            // Verify technique is correct
            assert_eq!(mov.technique, "hidden_single");

            // Verify value is in valid range.
            assert!(
                mov.value.get() >= 1 && mov.value.get() <= 9,
                "Move value {} should be in range 1-9",
                mov.value.get()
            );

            // Verify cell index is valid.
            assert!(
                mov.index < 81,
                "Move index {} should be in range 0-80",
                mov.index
            );

            // Verify the move is possible on the board.
            assert!(
                board.is_value_possible(mov.index, mov.value.get()),
                "Move at cell {} with value {} should be possible on the board",
                mov.index,
                mov.value.get()
            );

            // Verify it's not a naked single (cell should have multiple possibilities).
            let possible_count = board.count_possible(mov.index);
            assert!(
                possible_count > 1,
                "Cell {} should have multiple possibilities (has {}), not be a naked single",
                mov.index,
                possible_count
            );

            // Verify the value is actually one of the possible values.
            let possible_values = board.get_possible_values(mov.index);
            assert!(
                possible_values.contains(&mov.value.get()),
                "Value {} should be in possible values {:?} for cell {}",
                mov.value.get(),
                possible_values,
                mov.index
            );

            // Verify it's truly a hidden single: this value can only go in this cell
            // in at least one zone containing this cell.
            let mut found_zone_with_unique_placement = false;

            for &zone_index in board.get_zones_for_cell(mov.index).unwrap() {
                let zone = board.get_cells_for_zone(zone_index).unwrap();
                let mut cells_where_value_possible = Vec::new();

                for &cell_index in zone {
                    if board.get_value(cell_index).is_none()
                        && board.is_value_possible(cell_index, mov.value.get())
                    {
                        cells_where_value_possible.push(cell_index);
                    }
                }

                // Check if this zone has the value constrained to only this cell
                if cells_where_value_possible.len() == 1 {
                    assert_eq!(
                        cells_where_value_possible[0],
                        mov.index,
                        "The only cell where value {} is possible in zone {} should be cell {}",
                        mov.value.get(),
                        zone_index,
                        mov.index
                    );

                    found_zone_with_unique_placement = true;
                    break;
                }
            }

            assert!(
                found_zone_with_unique_placement,
                "Hidden single at cell {} with value {} should have at least one zone where it's the only possible placement",
                mov.index,
                mov.value.get()
            );
        }
    }
}
