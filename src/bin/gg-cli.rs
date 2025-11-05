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
    let board: SudokuBoard = args.puzzle.parse()?;
    let mut solver = SudokuDlxSolver::new();

    let limit = args
        .max_solutions
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
            println!("{solution}\n");
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
}
