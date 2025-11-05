//! Sudoku solver scaffolding.
//!
//! This module provides the foundational data structures required to build Sudoku solvers on
//! top of the Dancing Links (`dlx`) implementation. It mirrors the legacy C++ solver layout by
//! exposing Sudoku-specific graph metadata and a lightweight board representation.

use std::{
    borrow::Cow,
    collections::HashSet,
    fmt,
    str::FromStr,
};

/// Metadata describing the constraint graph for a Sudoku-like puzzle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneMetadata {
    pub num_cells: usize,
    pub num_values: usize,
    pub zones: Vec<Vec<usize>>,
    /// For each cell, the list of other cells that share at least one zone.
    pub neighbors_for_cell: Vec<Vec<usize>>,
    /// For each cell, the zones the cell participates in.
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
    pub fn classic() -> SudokuGraph {
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
    cells: [u8; SudokuGraph::NUM_CELLS],
}

impl SudokuBoard {
    /// Creates an empty board (all cells unset).
    pub fn empty() -> SudokuBoard {
        SudokuBoard {
            cells: [0; SudokuGraph::NUM_CELLS],
        }
    }

    /// Returns the stored digit at the provided cell index (0 for empty).
    pub fn value(&self, index: usize) -> u8 {
        assert!(index < SudokuGraph::NUM_CELLS);
        self.cells[index]
    }

    /// Sets a digit (1-9) or clears a cell by providing 0.
    pub fn set_value(&mut self, index: usize, value: u8) -> Result<(), String> {
        if index >= SudokuGraph::NUM_CELLS {
            return Err(format!("cell index {index} is out of bounds"));
        }
        if value > 9 {
            return Err(format!("value {value} is outside the allowed range 0-9"));
        }
        self.cells[index] = value;
        Ok(())
    }

    /// Removes any digit from the given cell.
    pub fn clear(&mut self, index: usize) {
        assert!(index < SudokuGraph::NUM_CELLS);
        self.cells[index] = 0;
    }

    /// Iterator over indices that currently have givens (non-zero values).
    pub fn given_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(idx, &value)| if value != 0 { Some(idx) } else { None })
    }

    /// Returns an owning string representation using '.' for empty cells.
    pub fn to_puzzle_string(&self) -> String {
        self.cells
            .iter()
            .map(|&value| match value {
                0 => '.',
                digit => (b'0' + digit) as char,
            })
            .collect()
    }
}

impl fmt::Display for SudokuBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..9 {
            if row != 0 {
                writeln!(f)?;
            }
            for col in 0..9 {
                let value = self.value(row * 9 + col);
                if value == 0 {
                    write!(f, ".")?;
                } else {
                    write!(f, "{value}")?;
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
        let trimmed: Cow<'_, str> = if s.len() == SudokuGraph::NUM_CELLS {
            Cow::Borrowed(s)
        } else {
            Cow::Owned(s.chars().filter(|c| !c.is_whitespace()).collect())
        };

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
                '1'..='9' => ch as u8 - b'0',
                '0' | '.' => 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_metadata_has_expected_geometry() {
        let graph = SudokuGraph::classic();
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
        assert_eq!(board.value(0), 5);
        assert_eq!(board.value(1), 3);
        assert_eq!(board.value(2), 0);
        assert_eq!(board.value(80), 9);

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
        assert!(board.set_value(10, 5).is_ok());
        assert_eq!(board.value(10), 5);

        assert!(board.set_value(10, 0).is_ok());
        assert_eq!(board.value(10), 0);

        let err = board.set_value(100, 1).unwrap_err();
        assert!(err.contains("out of bounds"));

        let err = board.set_value(0, 12).unwrap_err();
        assert!(err.contains("outside the allowed range"));
    }
}
