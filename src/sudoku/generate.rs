use std::time::Instant;

use rand::{Rng, seq::SliceRandom};

use super::{SudokuBoard, SudokuDlxSolver};
use crate::gamedef::GameDefinition;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SudokuGenerateError {
    /// No unique solution were found on this run of the generator. Try again.
    NoUniqueSolutions,

    // The number of clues to remove the board
    InvalidNumberOfRequestedRemovals,

    // The given Sudoku board is not fully filled.
    IncompleteSuokduBoard,
}

pub fn generate_solved_sudoku_board<R: Rng>(
    rng: &mut R,
) -> Result<SudokuBoard, SudokuGenerateError> {
    let mut solver = SudokuDlxSolver::new();

    let mut board = SudokuBoard::new();

    // Shuffle the cell indexes so that the board is filled in a random order.
    let indexes = {
        let mut v = (0..board.num_cells()).collect::<Vec<usize>>();
        v.shuffle(rng);
        v
    };

    for index in indexes {
        // Get a list of possible values at the current index and shuffle for randomness.
        let mut possible_values_for_cell = board.get_possible_values(index);
        possible_values_for_cell.shuffle(rng);

        // Try placing each possible value in the cell until there is at least one valid solution.
        let mut placed_value = false;
        for possible_value_for_cell in possible_values_for_cell {
            board
                .set_value(index, possible_value_for_cell)
                .expect("value was already known to be possible");

            let solutions = solver.solve_with_limit(&board, Some(2));
            if !solutions.is_empty() {
                // If there is at least one possible solution with this value at this cell, then
                // this placement was successful so break and continue with next cell.
                placed_value = true;
                break;
            }

            // If the placement failed because there are no solutions, then clear the cell
            // and continue with trying possible values.
            board
                .reset_value(index)
                .expect("restore cell value to original value");
        }

        if !placed_value {
            return Err(SudokuGenerateError::NoUniqueSolutions);
        }
    }

    Ok(board)
}

/// Generate a Sudoku puzzle by attempting to remove up to the specified number of clues
/// from an existing valid solution. The caller should check the returned `SudokuBoard`
/// to determine the number of given values actually removed from the board.
pub fn remove_given_values_from_board<R: Rng>(
    mut board: SudokuBoard,
    max_num_values_to_remove: usize,
    rng: &mut R,
) -> Result<SudokuBoard, SudokuGenerateError> {
    let mut solver = SudokuDlxSolver::new();

    if max_num_values_to_remove == 0 || max_num_values_to_remove >= board.num_cells() {
        return Err(SudokuGenerateError::InvalidNumberOfRequestedRemovals);
    }

    log::debug!("Removing {max_num_values_to_remove} from board.");

    // Shuffle the cell indexes so that the clues are removed in a random order.
    let indexes = {
        let mut v = (0..board.num_cells()).collect::<Vec<usize>>();
        v.shuffle(rng);
        v
    };

    let mut num_values_removed = 0usize;
    for index in indexes {
        let prior_value = board
            .get_value(index)
            .ok_or(SudokuGenerateError::IncompleteSuokduBoard)?;
        board.reset_value(index).unwrap();

        log::debug!("Trying removal of value {prior_value} in cell index {index}.");

        let start_time = Instant::now();
        let num_solutions = solver.solve_with_limit(&board, Some(2));
        let uniqueness_check_duration = start_time.elapsed();

        if num_solutions.len() == 1 {
            // If the board still has a unique solution, even after removing this clue, then
            // the placement was successful.
            num_values_removed += 1;
            log::debug!(
                "Removal succeeded. num_values_removed={num_values_removed}. (Duration: {uniqueness_check_duration:?})"
            );
            if num_values_removed >= max_num_values_to_remove {
                break;
            }
        } else {
            // Removing this clue resulted in an unsolveable board or a board with multiple solutions.
            // Put the value back and continue with trying other cells.
            log::debug!(
                "Removal failed due to no unique solution existing. (Duration: {uniqueness_check_duration:?})"
            );
            board
                .set_value(index, prior_value)
                .expect("restoring original value");
        }
    }

    Ok(board)
}

pub fn generate_sudoku_board_with_rng<R: Rng>(
    max_num_values_to_remove: usize,
    rng: &mut R,
) -> Result<SudokuBoard, SudokuGenerateError> {
    let solved_board = generate_solved_sudoku_board(rng)?;
    let puzzle_board = remove_given_values_from_board(solved_board, max_num_values_to_remove, rng)?;
    Ok(puzzle_board)
}

pub fn generate_sudoku_puzzle(
    max_num_values_to_remove: usize,
) -> Result<SudokuBoard, SudokuGenerateError> {
    let mut rng = rand::rng();
    generate_sudoku_board_with_rng(max_num_values_to_remove, &mut rng)
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn generator_successfully_generates_a_unique_sudoku_puzzle() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        let num_clues_to_remove = 40;
        let puzzle = generate_sudoku_board_with_rng(num_clues_to_remove, &mut rng)
            .expect("generator should produce a valid puzzle");

        assert_eq!(
            puzzle.given_indices().count(),
            puzzle.num_cells() - num_clues_to_remove
        );

        let mut solver = SudokuDlxSolver::new();
        let solutions = solver.solve_with_limit(&puzzle, Some(2));
        assert_eq!(solutions.len(), 1, "generated puzzle must be unique");
    }
}
