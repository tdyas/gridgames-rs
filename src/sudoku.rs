//! Suokdu solver

use std::collections::HashSet;

struct ZoneMetadata {
    pub num_cells: usize,
    pub num_values: usize,
    pub zones: Vec<Vec<usize>>,
    
    /// For each cell, a list of other cells in the same zones as the specified cell.
    pub neighbors_for_cell: Vec<Vec<usize>>,
    /// For each cell, a list of zones encompassing that cell.
    pub zones_for_cell: Vec<Vec<usize>>,
}

pub struct ZoneMetadata {
    pub fn new(num_cells: usize, num_values: usize, zones: Vec<Vec<usize>>) -> ZoneMetadata {
        let mut cell_sets: Vec<HashSet> = Vec::with_capacity(num_cells);
        for _ in 0..num_cells {
            cell_sets.push(HashSet::new())
        }

        for zone in &zones {
            for i in &zone {
                for j in &zone {
                    if i == j {
                        continue;
                    }
                    cell_sets[i].add(*j);
                }
            }
        }
    }
}