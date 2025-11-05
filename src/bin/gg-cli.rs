use clap::{Args, Parser, Subcommand};
use gridgames_rs::sudoku::{SudokuBoard, SudokuDlxSolver};
use std::process;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        MainCommand::Sudoku(sudoku_command) => match sudoku_command.command {
            SudokuSubcommand::Solve(args) => execute_sudoku_solve(args),
        },
    }
}

fn execute_sudoku_solve(args: SudokuSolveArgs) -> Result<(), String> {
    let SudokuSolveArgs {
        puzzle,
        max_solutions,
        show_zones,
    } = args;

    let board: SudokuBoard = puzzle.parse()?;
    let mut solver = SudokuDlxSolver::new();

    let limit = max_solutions
        .map(|value| {
            if value == 0 {
                Err("max-solutions must be greater than zero".to_string())
            } else {
                Ok(value as usize)
            }
        })
        .transpose()?;

    let solutions = solver.solve_with_limit(&board, limit);
    if solutions.is_empty() {
        println!("No solutions found.");
    } else {
        for (idx, solution) in solutions.iter().enumerate() {
            println!("Solution {}:", idx + 1);
            if show_zones {
                println!("{}", format_board_with_zones(solution));
            } else {
                println!("{solution}");
            }
            println!();
        }
        println!("Total solutions returned: {}", solutions.len());
    }

    Ok(())
}

#[derive(Parser)]
#[command(name = "gg-cli", version, about = "GridGames CLI tools")]
struct Cli {
    #[command(subcommand)]
    command: MainCommand,
}

#[derive(Subcommand)]
enum MainCommand {
    /// Sudoku-related commands
    Sudoku(SudokuCommand),
}

#[derive(Args)]
struct SudokuCommand {
    #[command(subcommand)]
    command: SudokuSubcommand,
}

#[derive(Subcommand)]
enum SudokuSubcommand {
    /// Solve a Sudoku puzzle using the DLX solver
    Solve(SudokuSolveArgs),
}

#[derive(Args)]
struct SudokuSolveArgs {
    /// 81-character puzzle string using digits and '.' for empty cells.
    #[arg()]
    puzzle: String,

    /// Maximum number of solutions to return (default: 2 per library API)
    #[arg(long)]
    max_solutions: Option<u32>,

    /// Print solution boards with row/column/box separators.
    #[arg(long)]
    show_zones: bool,
}

fn format_board_with_zones(board: &SudokuBoard) -> String {
    let mut lines = Vec::new();
    let mut last_width = 0usize;

    for row in 0..9 {
        let mut row_chars = ['.'; 9];
        for col in 0..9 {
            let idx = row * 9 + col;
            row_chars[col] = board
                .value(idx)
                .map(|digit| char::from(b'0' + digit))
                .unwrap_or('.');
        }

        let row_line = format!(
            "{} {} {} | {} {} {} | {} {} {}",
            row_chars[0],
            row_chars[1],
            row_chars[2],
            row_chars[3],
            row_chars[4],
            row_chars[5],
            row_chars[6],
            row_chars[7],
            row_chars[8]
        );

        if row % 3 == 0 {
            lines.push("-".repeat(row_line.len()));
        }

        last_width = row_line.len();
        lines.push(row_line);
    }

    if last_width == 0 && !lines.is_empty() {
        last_width = lines[0].len();
    }
    if last_width > 0 {
        lines.push("-".repeat(last_width));
    }

    lines.join("\n")
}
