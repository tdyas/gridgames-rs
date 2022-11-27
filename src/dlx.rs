//! Knuth's DLX "Dancing Links" algorithm
//!
//! This module implements Donald Knuth's "Algorhtm X" for solving the "exact cover" problem
//! using the ["Dancing Links" method](https://arxiv.org/abs/cs/0011047). The exact cover
//! problem allows for solving constraint satisfaction problems including solving
//! Sudoku boards. Algorithm X with Dancing Links is also called "DLX."
//!
//! My original implementation was in C++ (in the GridGames repository for my released iOS
//! "Shoal Sudoku" game) and used doubly-linked lists with heap allocated nodes for the
//! sparse matrix.
//!
//! This Rust version is loosely inspired by https://ferrous-systems.com/blog/dlx-in-rust/
//! and Knuth's paper which both use Entity Component System (ECS) form, i.e. side tables, which is
//! much more friendly to implement in Safe Rust. Although I have used more descriptive names for
//! the side tables then the single letter names used in Knuth's paper.

use std::{fmt, ops};

/// A node represents a `true` value in the sparse matrix used to represent rows and columns for the DLX
/// algorithm. Since this module uses ECS representation for nodes, `Node` is just a new-typed index into
/// the side tables where the node entity's data is actually stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Node(usize);

/// This is a singleton node for the column header row. Knuth calls it `h` in his paper, but I find
/// that this name is much more descriptive.
///
/// Note: Some of the data structures will have a dummy first node to account for the fact that the root node
/// exists and has index 0. This allows indexing into those side tables using a `Node` without having to
/// adjust the index by 1.
const ROOT_NODE: Node = Node(0);

/// DLX encapsulates state for applying the DLX algorithm to solve an exact cover problem.
pub struct DLX {
    /// Number of columns in the matrix.
    num_columns: usize,

    /// Linked list data for links between nodes in the same column. This is what Knuth calls `U` and `D`
    /// in the paper.
    column_links: NodeLinks,

    /// Linked list data for links across nodes in the same row. This is what Knuth cals `L` and `R`
    /// in the paper.
    row_links: NodeLinks,

    /// Number of nodes in each column. Indexed by the column's node. Thus, there is a dummy element
    /// at index 0 to account for ROOT_NODE.
    ///
    /// This is what Knuth calls `S` in the paper.
    column_sizes: Vec<usize>,

    /// Column header nodes. Each row where this column has a true value will be part of the `column_links` list.
    /// Note: This list does *not* have a dummy node for `ROOT_NODE` since it is intended to be indexed by
    /// column number.
    column_nodes: Vec<Node>,

    /// Map each node to its respective column node. This is what Knuth calls `C` in the paper.
    column_node_for_node: Vec<Node>,

    /// Node ranges for each row. This allows turning a list of solution nodes back into a list of row indices
    /// which will be more meaningful to the caller.
    row_node_ranges: Vec<ops::RangeInclusive<Node>>,

    /// List of nodes currently in the tenetative solution.
    proposed_solution_row_nodes: Vec<Node>,
}

impl DLX {
    /// Create a new DLX for solving an exact cover problem with the given number of columns. (The number of
    /// columns cannot be changed after a DLX is initialized.)
    ///
    /// Rows can then be added using `add_row` and then search for solutions using `solve`.
    pub fn new(num_columns: usize) -> Self {
        let mut dlx = DLX {
            num_columns,
            column_links: NodeLinks::new(),
            row_links: NodeLinks::new(),
            column_sizes: Vec::new(),
            column_nodes: Vec::new(),
            column_node_for_node: Vec::new(),
            row_node_ranges: Vec::new(),
            proposed_solution_row_nodes: Vec::new(),
        };

        // Allocate the root node.
        let root_node = dlx.alloc();
        assert_eq!(root_node, ROOT_NODE);
        dlx.column_sizes.push(0);

        // Setup each column by allocating a node per column and link into the header row.
        for _ in 0..num_columns {
            let column_header_node = dlx.alloc();
            dlx.column_nodes.push(column_header_node);
            dlx.column_sizes.push(0);

            // Link this column's header node into the header row.
            dlx.row_links
                .insert(dlx.row_links[ROOT_NODE].previous, column_header_node);
        }

        dlx
    }

    /// Allocate a new node using arena-allocation.
    fn alloc(&mut self) -> Node {
        let new_node = self.row_links.alloc();
        let new_node_2 = self.column_links.alloc();
        assert_eq!(new_node, new_node_2);

        self.column_node_for_node.push(ROOT_NODE);
        assert_eq!(self.column_node_for_node.len(), new_node.0 + 1);

        new_node
    }

    /// Append a new row to the DLX given a slice of bool values denoting which
    /// columns are covered by the row.
    pub fn push_row(&mut self, row_values: &[bool]) {
        assert_eq!(row_values.len(), self.num_columns);

        let mut first_node = None;
        let mut last_node = None;

        for col_num in 0..self.num_columns {
            // Skip false values since we are using a sparse matrix of true values.
            if !row_values[col_num] {
                continue;
            }

            // Increment the size of this column. This will be used to optimize for columns with
            // fewer nodes to reduce the algorithm's branching factor.
            self.column_sizes[col_num + 1] += 1;

            // Allocate the node and link it into its column.
            let node = self.alloc();
            let col_node = self.column_nodes[col_num];
            self.column_node_for_node[node.0] = col_node;
            self.column_links
                .insert(self.column_links[col_node].previous, node);

            // Link this node into the row if it is not the first node in the row.
            if let Some(last_node) = last_node {
                self.row_links.insert(last_node, node);
            } else {
                first_node = Some(node);
            }
            last_node = Some(node);
        }

        if let (Some(first_node), Some(last_node)) = (first_node, last_node) {
            self.row_node_ranges
                .push(ops::RangeInclusive::new(first_node, last_node));
        }
    }

    /// Internal helper for recursively searching for solutions.
    fn search<F>(&mut self, found_solution: &mut F)
    where
        F: FnMut(Vec<usize>),
    {
        // If all columns are covered, then we have found a solution.
        if self.row_links[ROOT_NODE].next == ROOT_NODE {
            let solution = self.convert_row_nodes_to_row_indices(&self.proposed_solution_row_nodes);
            found_solution(solution);
            return;
        }

        // Choose the column from the remaining columns with the smallest "branching factor" to minimize work
        // done by the algorithm. This will be the column with the fewest number of rows with `true` values.
        let selected_col_node = {
            let mut selected_node = None;
            let mut selected_size = usize::MAX;

            let mut node = self.row_links[ROOT_NODE].next;
            while node != ROOT_NODE {
                if self.column_sizes[node] < selected_size {
                    selected_node = Some(node);
                    selected_size = self.column_sizes[node];
                }
                node = self.row_links[node].next;
            }

            selected_node.expect("at least one column should have existed")
        };

        // Cover the selected column.
        self.cover_column(selected_col_node);

        // Loop through each row in the selected column and tentatively add it to the proposed solution.
        // Before recursing, cover any other columns that each of those those rows are in.
        let mut current_node = self.column_links[selected_col_node].next;
        while current_node != selected_col_node {
            self.proposed_solution_row_nodes.push(current_node);

            // Cover every other column covered by the current row.
            let mut node_in_current_row = self.row_links[current_node].next;
            while node_in_current_row != current_node {
                let col_node = self.column_node_for_node[node_in_current_row];
                self.cover_column(col_node);

                node_in_current_row = self.row_links[node_in_current_row].next;
            }

            // Continue searching for solutions witin the remaining rows.
            self.search(found_solution);

            // Remove the current row from the list of proposed solution rows.
            self.proposed_solution_row_nodes.pop();

            // Uncover every other column in this row now that it is no longer a potential solution row.
            // Note: This must be done in the reverse order from how the coluns were covered. Thus "previous"
            // links are used to traverse instead of "next" links.
            let mut node_in_current_row = self.row_links[current_node].previous;
            while node_in_current_row != current_node {
                let col_node = self.column_node_for_node[node_in_current_row];
                self.uncover_column(col_node);

                node_in_current_row = self.row_links[node_in_current_row].previous;
            }

            // Advance to the next row.
            current_node = self.column_links[current_node].next;
        }

        // Uncover the selected column.
        self.uncover_column(selected_col_node);
    }

    fn cover_column(&mut self, col_node: Node) {
        assert!(col_node.0 <= self.column_nodes.len());

        // Unlink this column from the header row.
        self.row_links.unlink(col_node);

        // Then traverse each row in the column ...
        let mut node = self.column_links[col_node].next;
        while node != col_node {
            // ... and remove that row from any other columns that it is in.
            let mut node_in_this_row = self.row_links[node].next;
            while node_in_this_row != node {
                // Unlink this row from from this other column.
                self.column_links.unlink(node_in_this_row);

                let this_col_node = self.column_node_for_node[node_in_this_row];
                self.column_sizes[this_col_node] =
                    self.column_sizes[this_col_node].saturating_sub(1);
                node_in_this_row = self.row_links[node_in_this_row].next;
            }

            node = self.column_links[node].next;
        }
    }

    fn uncover_column(&mut self, col_node: Node) {
        assert!(col_node.0 <= self.column_nodes.len());

        // Traverse each row in the column ...
        let mut node = self.column_links[col_node].previous;
        while node != col_node {
            // ... and link the row back into any other columns that it is in.
            let mut node_in_this_row = self.row_links[node].next;
            while node_in_this_row != node {
                // Link this row from from this other column.
                self.column_links.link(node_in_this_row);

                let this_col_node = self.column_node_for_node[node_in_this_row];
                self.column_sizes[this_col_node.0] += 1;

                node_in_this_row = self.row_links[node_in_this_row].next;
            }

            node = self.column_links[node].previous;
        }

        // Link this column back into the header row.
        self.row_links.link(col_node);
    }

    fn convert_row_nodes_to_row_indices(&self, row_nodes: &[Node]) -> Vec<usize> {
        let mut result = Vec::new();

        let mut i = 0;
        while i < row_nodes.len() {
            let solution_row_node = row_nodes[i];
            for (j, range) in self.row_node_ranges.iter().enumerate() {
                if range.contains(&solution_row_node) {
                    result.push(j);
                }
            }
            i += 1;
        }

        result.sort();
        result
    }

    /// Attempt to find solutions to the exact cover problem configured in this `DLX`. Each found solution will
    /// trigger a call to the given closure.
    pub fn solve<F>(&mut self, mut found_solution: F)
    where
        F: FnMut(Vec<usize>),
    {
        self.search(&mut found_solution);
    }
}

// Neat trick from the Ferrous blog: Index `Vec` by a `Node` so that we do not have to violate its status as a newtype.

impl<T> ops::Index<Node> for Vec<T> {
    type Output = T;
    fn index(&self, index: Node) -> &Self::Output {
        &self[index.0]
    }
}

impl<T> ops::IndexMut<Node> for Vec<T> {
    fn index_mut(&mut self, index: Node) -> &mut Self::Output {
        &mut self[index.0]
    }
}

/// Represents a single link in a greater linked list for a row or column.
#[derive(Debug, PartialEq, Eq)]
struct Link {
    previous: Node,
    next: Node,
}

/// Represents all of the links between nodes in the DLX sparse matrix in one "direction". That is,
/// one instance of `NodeLinks` represents the links between nodes in the same column and another
/// instance represents links between nodes in the same row.
struct NodeLinks {
    data: Vec<Link>,
}

impl NodeLinks {
    pub fn new() -> Self {
        NodeLinks { data: Vec::new() }
    }

    /// Arena allocate the next node in the links data.
    pub fn alloc(&mut self) -> Node {
        let node = Node(self.data.len());
        self.data.push(Link {
            previous: node,
            next: node,
        });
        node
    }

    // Insert `b` into a<->c to produce a<->b<->c
    pub fn insert(&mut self, a: Node, b: Node) {
        let c = self[a].next;

        // a <- b and b -> c
        self[b].previous = a;
        self[b].next = c;

        // a -> b
        self[a].next = b;

        // b <- c
        self[c].previous = b;
    }

    /// Unlink node `b` from the linked list a<->b<->c without disturbing the links out of `b`
    /// so `b` can be restored to the list by calling `link` later.
    pub fn unlink(&mut self, b: Node) {
        let b_previous = self[b].previous;
        let b_next = self[b].next;
        self[b_previous].next = b_next;
        self[b_next].previous = b_previous;
    }

    /// Links node `b` back into the linked list a<->c to produce a<->b<->c using the undisturbed links out of `b`.
    /// This is the heart of the "Dancing Links" part of Knuth's algorithm.
    pub fn link(&mut self, b: Node) {
        let b_previous = self[b].previous;
        let b_next = self[b].next;
        self[b_previous].next = b;
        self[b_next].previous = b;
    }
}

impl ops::Index<Node> for NodeLinks {
    type Output = Link;
    fn index(&self, index: Node) -> &Self::Output {
        &self.data[index.0]
    }
}

impl ops::IndexMut<Node> for NodeLinks {
    fn index_mut(&mut self, index: Node) -> &mut Self::Output {
        &mut self.data[index.0]
    }
}

impl fmt::Debug for NodeLinks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeLinks(")?;
        let mut comma = false;
        for (i, Link { previous, next }) in self.data.iter().enumerate() {
            if comma {
                write!(f, ", ")?;
            }
            comma = true;

            write!(f, "{i}:{:?}-{:?}", *previous, *next)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Link, Node, NodeLinks, DLX};

    #[test]
    fn node_links_basic_test() {
        let mut node_links = NodeLinks::new();
        let a = node_links.alloc();
        let b = node_links.alloc();
        let c = node_links.alloc();

        assert_eq!(
            node_links[a],
            Link {
                previous: Node(0),
                next: Node(0)
            }
        );
        assert_eq!(
            node_links[b],
            Link {
                previous: Node(1),
                next: Node(1)
            }
        );
        assert_eq!(
            node_links[c],
            Link {
                previous: Node(2),
                next: Node(2)
            }
        );

        node_links.insert(a, b);

        assert_eq!(
            node_links[a],
            Link {
                previous: Node(1),
                next: Node(1)
            }
        );
        assert_eq!(
            node_links[b],
            Link {
                previous: Node(0),
                next: Node(0)
            }
        );
        assert_eq!(
            node_links[c],
            Link {
                previous: Node(2),
                next: Node(2)
            }
        );

        node_links.insert(b, c);

        assert_eq!(
            node_links[a],
            Link {
                previous: Node(2),
                next: Node(1)
            }
        );
        assert_eq!(
            node_links[b],
            Link {
                previous: Node(0),
                next: Node(2)
            }
        );
        assert_eq!(
            node_links[c],
            Link {
                previous: Node(1),
                next: Node(0)
            }
        );

        node_links.unlink(b);

        assert_eq!(
            node_links[a],
            Link {
                previous: Node(2),
                next: Node(2)
            }
        );
        assert_eq!(
            node_links[b],
            Link {
                previous: Node(0),
                next: Node(2)
            }
        );
        assert_eq!(
            node_links[c],
            Link {
                previous: Node(0),
                next: Node(0)
            }
        );

        node_links.link(b);

        assert_eq!(
            node_links[a],
            Link {
                previous: Node(2),
                next: Node(1)
            }
        );
        assert_eq!(
            node_links[b],
            Link {
                previous: Node(0),
                next: Node(2)
            }
        );
        assert_eq!(
            node_links[c],
            Link {
                previous: Node(1),
                next: Node(0)
            }
        );
    }

    #[test]
    fn small_problem() {
        let mut dlx = DLX::new(3);
        dlx.push_row(&[true, false, true]);
        dlx.push_row(&[false, true, false]);
        dlx.push_row(&[true, false, false]);
        dlx.push_row(&[false, false, true]);
        let mut solutions = Vec::new();
        dlx.solve(|solution| {
            solutions.push(solution);
        });
        assert_eq!(solutions.len(), 2);
        assert_eq!(solutions, vec![vec![0, 1], vec![1, 2, 3],]);
    }

    #[test]
    fn example_from_knuth_paper() {
        // This problem comes from the Kunth Dancing Links paper.
        let mut dlx = DLX::new(7);
        dlx.push_row(&[false, false, true, false, true, true, false]);
        dlx.push_row(&[true, false, false, true, false, false, true]);
        dlx.push_row(&[false, true, true, false, false, true, false]);
        dlx.push_row(&[true, false, false, true, false, false, false]);
        dlx.push_row(&[false, true, false, false, false, false, true]);
        dlx.push_row(&[false, false, false, true, true, false, true]);
        let mut solutions = Vec::new();
        dlx.solve(|solution| {
            solutions.push(solution);
        });
        assert_eq!(solutions, vec![vec![0, 3, 4]]);
    }

    /// Solve a single instance of the four-column version of DLX using `seed` as the values for each row.
    fn solve_seed(seed: u16) -> bool {
        let mut rows = [0u16; 4];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = (seed >> (i * 4)) & 0b1111;
            if *row == 0 {
                // TODO: Why return in this case?
                return false;
            }
        }

        let brute_force = {
            let mut n_solutions = 0;
            for mask in 0..=0b1111 {
                let mut or = 0;
                let mut n_ones = 0;
                for (i, &row) in rows.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        or |= row;
                        n_ones += row.count_ones()
                    }
                }
                if or == 0b1111 && n_ones == 4 {
                    n_solutions += 1;
                }
            }
            n_solutions
        };

        let dlx = {
            let mut m = DLX::new(4);
            for row_bits in rows.iter() {
                let mut row = [false; 4];
                for i in 0..4 {
                    row[i] = row_bits & (1 << i) != 0;
                }
                m.push_row(&row);
            }
            let mut count = 0;
            m.solve(|_solution| count += 1);
            count
        };

        assert_eq!(brute_force, dlx);
        true
    }

    /// Test the DLX algorithm by solving all four-column cases. By iterating over a u16 and decomposing it into
    /// into its component 4-bit nibbles, we have every possible combination of rows for four columns.
    #[test]
    fn brute_force() {
        for seed in u16::MIN..=u16::MAX {
            solve_seed(seed);
        }
    }
}
