use std::collections::HashSet;

/// Errors returned from various `GameDefinition` methods.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GameDefinitionError {
    /// The given zone index was out of bounds.
    InvalidZoneIndex,

    /// The given cell index was out of bounds.
    InvalidCellIndex,
}

/// Definition of a "grid game" including zone metadata.
pub trait GameDefinition {
    /// Return the number of cells in this game.
    fn num_cells(&self) -> usize;

    /// Return the number of values in this game.
    fn num_values(&self) -> u8;

    /// Return the number of zones. A zone is a set of cells where a value may only appear once.
    fn num_zones(&self) -> usize;

    /// Return the cells for a given zone.
    fn get_cells_for_zone(&self, zone_index: usize) -> Result<&[usize], GameDefinitionError>;

    /// Return the "neighbors" of the given cell. A neighbor is a cell which shares at least
    /// one other zone with the given cell.
    fn get_neighbors_for_cell(&self, cell_index: usize) -> Result<&[usize], GameDefinitionError>;

    /// Return the zones in which the cell lies.
    fn get_zones_for_cell(&self, cell_index: usize) -> Result<&[usize], GameDefinitionError>;
}

#[derive(Clone, Debug)]
pub struct GenericGameDefinition {
    // Total number of cells in the game.
    num_cells: usize,

    // Number of values allowed in the same (e.g, for Sudoku 1-9 are the values).
    num_values: u8,

    // Define each zone by the cell indices for that zone.
    zones: Vec<Vec<usize>>,

    /// For each cell, the list of other cells that share at least one zone.
    neighbors_for_cell: Vec<Vec<usize>>,

    /// For each cell, the zones the cell is a member of.
    zones_for_cell: Vec<Vec<usize>>,
}

impl GenericGameDefinition {
    /// Builds derived neighbor and zone lookup tables for the provided zone definitions.
    pub fn new(num_cells: usize, num_values: u8, zones: Vec<Vec<usize>>) -> GenericGameDefinition {
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

        GenericGameDefinition {
            num_cells,
            num_values,
            zones,
            neighbors_for_cell,
            zones_for_cell,
        }
    }
}

impl GameDefinition for GenericGameDefinition {
    fn num_cells(&self) -> usize {
        self.num_cells
    }

    fn num_values(&self) -> u8 {
        self.num_values
    }

    fn num_zones(&self) -> usize {
        self.zones.len()
    }

    fn get_cells_for_zone(&self, zone_index: usize) -> Result<&[usize], GameDefinitionError> {
        self.zones
            .get(zone_index)
            .map(|v| v.as_ref())
            .ok_or(GameDefinitionError::InvalidZoneIndex)
    }

    fn get_neighbors_for_cell(&self, cell_index: usize) -> Result<&[usize], GameDefinitionError> {
        self.neighbors_for_cell
            .get(cell_index)
            .map(|v| v.as_ref())
            .ok_or(GameDefinitionError::InvalidCellIndex)
    }

    fn get_zones_for_cell(&self, cell_index: usize) -> Result<&[usize], GameDefinitionError> {
        self.zones_for_cell
            .get(cell_index)
            .map(|v| v.as_ref())
            .ok_or(GameDefinitionError::InvalidCellIndex)
    }
}
