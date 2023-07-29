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

    let smir = ssa::mir::StableDisplayMir::from(mir.clone());
    println!("{0:=>18} OPT MIR {0:=>18}", "");
    println!("{smir}");

    println!("digraph basic_blocks {{");
    for (&block_id, block) in mir.blocks() {
        match &block.term {
            ssa::mir::Terminator::Jump(next) => {
                println!("{block_id} -> {}", next.id);
            }
            ssa::mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                println!("{block_id} -> {}", if_true.id);
                println!("{block_id} -> {}", if_false.id);
            }
            ssa::mir::Terminator::ProgramExit => (),
        }
    }
    println!("}}");

    let mir = ssa::to_ssa::to_ssa(&mir);

    let smir = ssa::mir::StableDisplayMir::from(mir.clone());
    println!("{0:=>20} SSA {0:=>20}", "");
    println!("{smir}");

    Ok(())
}
