use std::process;

use clap::{Args, Parser, Subcommand, ValueEnum};
use gridgames_rs::{
    gamedef::GameDefinition,
    sudoku::{
        SudokuBoard, SudokuDlxSolver,
        generate::{generate_solved_sudoku_board, remove_given_values_from_board},
    },
};

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
    /// Solve a Sudoku puzzle using the DLX solver.
    Solve(SudokuSolveArgs),

    /// Generate a unique Sudoku puzzle by removing clues from a solved board.
    Generate(SudokuGenerateArgs),

    /// Generate a solved Sudoku board with no clues removed.
    MakeSolvedBoard,
}

#[derive(Args)]
struct SudokuSolveArgs {
    /// 81-character puzzle string using digits and '.' for empty cells.
    #[arg()]
    puzzle: String,

    /// Maximum number of solutions to return (default: 2 per library API)
    #[arg(long)]
    max_solutions: Option<usize>,

    /// Print solution boards with row/column/box separators.
    #[arg(long)]
    show_zones: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    // Pretty print the Sudoku puzzle.
    PrettyPrint,

    // Output a JSON blob with the Sudoku puzzle and the solved version.
    Json,
}

#[derive(Args)]
struct SudokuGenerateArgs {
    // Generate a solved board only without removing clues.
    #[arg(long)]
    show_solved_board: bool,

    /// Number of clues to remove from a solved board (must be < 81).
    #[arg(short = 'v', long, value_name = "COUNT", required = true)]
    max_values_to_remove: usize,

    /// Use a precomputed solved board instead of computing a new one.
    #[arg(long)]
    solved_board: Option<String>,

    /// Output format for displaying the computed puzzle.
    #[arg(short, long, value_enum, default_value_t = OutputFormat::PrettyPrint)]
    output: OutputFormat,
}

fn main() {
    env_logger::init();
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        MainCommand::Sudoku(sudoku_command) => match sudoku_command.command {
            SudokuSubcommand::Solve(args) => sudoku_solve(args),
            SudokuSubcommand::Generate(args) => sudoku_generate(args),
            SudokuSubcommand::MakeSolvedBoard => sudoku_make_solved_board(),
        },
    }
}

fn sudoku_solve(args: SudokuSolveArgs) -> Result<(), String> {
    let SudokuSolveArgs {
        puzzle,
        max_solutions,
        show_zones,
    } = args;

    let board = SudokuBoard::from_puzzle_str(&puzzle)?;
    let mut solver = SudokuDlxSolver::new();

    let limit = max_solutions
        .map(|value| {
            if value == 0 {
                Err("max-solutions must be greater than zero".to_string())
            } else {
                Ok(value)
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
                println!("{solution:?}");
            }
            println!();
        }
        println!("Total solutions returned: {}", solutions.len());
    }

    Ok(())
}

fn sudoku_generate(args: SudokuGenerateArgs) -> Result<(), String> {
    let mut rng = rand::rng();

    let SudokuGenerateArgs {
        show_solved_board,
        max_values_to_remove,
        solved_board,
        output,
    } = args;

    let solved_board = if let Some(solved_board) = solved_board {
        SudokuBoard::from_puzzle_str(&solved_board)
            .map_err(|err| format!("Error while converting provided solved board: {err:?}"))?
    } else {
        generate_solved_sudoku_board(&mut rng)
            .map_err(|err| format!("Error while generating a new solved board: {err:?}"))?
    };

    if show_solved_board {
        println!("{}", format_board_with_zones(&solved_board));
    }

    if max_values_to_remove >= solved_board.num_cells() {
        return Err(format!(
            "clues-to-remove must be between 0 and {}",
            solved_board.num_cells() - 1
        ));
    }

    let puzzle =
        remove_given_values_from_board(solved_board.clone(), max_values_to_remove, &mut rng)
            .map_err(|err| format!("Error while removoing values from board: {err:?}"))?;

    let actual_num_clues_remaining = puzzle.given_indices().count();

    match output {
        OutputFormat::PrettyPrint => {
            println!(
                "Generated puzzle (removed {} clues, {} given values):",
                puzzle.num_cells() - actual_num_clues_remaining,
                actual_num_clues_remaining,
            );
            println!("{}", format_board_with_zones(&puzzle));
        }
        OutputFormat::Json => {
            println!(
                "{{\"solved\":\"{}\",\"puzzle\":\"{}\"}}",
                solved_board.to_puzzle_string(),
                puzzle.to_puzzle_string()
            )
        }
    }

    Ok(())
}

fn sudoku_make_solved_board() -> Result<(), String> {
    let mut rng = rand::rng();
    let solved_board = generate_solved_sudoku_board(&mut rng)
        .map_err(|err| format!("Error while generating solved board: {err:?}"))?;
    println!("{}", format_board_with_zones(&solved_board));
    Ok(())
}

fn format_board_with_zones(board: &SudokuBoard) -> String {
    let mut lines = Vec::new();
    let mut last_width = 0usize;

    for row in 0..9 {
        let mut row_chars = ['.'; 9];
        for col in 0..9 {
            let idx = row * 9 + col;
            row_chars[col] = board
                .get_value(idx)
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
