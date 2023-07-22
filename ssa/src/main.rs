use std::path::PathBuf;

use colorz::mode::{Mode, Stream};
use eyre::Context;

#[derive(clap::Parser)]
#[clap(rename_all = "kebab-case")]
struct Args {
    /// Selects the coloring mode for this run,
    /// must be one of "always", "never", "detect" case insensitive
    #[clap(long)]
    color_mode: Option<Mode>,

    input: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args: Args = clap::Parser::parse();

    if let Some(mode) = args.color_mode.or(Mode::from_env()) {
        colorz::mode::set_coloring_mode(mode);
    }

    colorz::mode::set_default_stream(Stream::Stdout);

    colorz_eyre::install()?;

    let input = std::fs::read(&args.input)
        .with_context(|| format!("Could not read file {}", args.input.display()))?;

    let syn = ssa::syntax::Syntax::from_bytes(&input)?;

    dbg!(&syn);

    println!("{syn}");

    Ok(())
}
