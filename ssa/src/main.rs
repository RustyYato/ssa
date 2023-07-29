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

    println!("{syn}\n");

    let enc = ssa::to_mir::Encoder::new();
    let mut mir = enc.encode(&syn)?;

    ssa::mir_opts::clean_up_jumps(&mut mir);

    println!("{0:=>18} OPT MIR {0:=>18}", "");
    println!("{mir}");

    let mir = ssa::to_ssa::to_ssa(&mir);

    let mir = ssa::mir::StableDisplayMir::from(mir);

    println!("{0:=>20} SSA {0:=>20}", "");
    println!("{mir}");

    Ok(())
}
