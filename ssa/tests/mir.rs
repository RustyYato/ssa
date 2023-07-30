#[macro_use]
mod mir {
    #[macro_use]
    pub mod harness;
}

test_source! {
    name: "to-mir",
    test_corpus: "tests/mir/pass",
    path_matcher: |path| matches_extension(path, "lisp"),
    expected_path: |path| {
        path.parent().unwrap().join("parse").join(path.file_name().unwrap())
            .with_extension("mir")
    },
    test_runner: |test| {
        let input = std::fs::read(&test.path)?;

        let syntax = match ssa::syntax::Syntax::from_bytes(&input) {
            Err(err) => return Ok(MaybeIgnore::Failed(format!("syntax error: {err}"))),
            Ok(x) => x,
        };

        match ssa::to_mir::Encoder::new().encode(&syntax) {
            Err(err) => Ok(MaybeIgnore::Failed(format!("encoding error: {err}"))),
            Ok(mir) => {
                let mir = ssa::mir::StableDisplayMir::from(mir);
                Ok(MaybeIgnore::Ran(format!("{mir}")))
            }
        }
    }
}

test_source! {
    name: "to-mir",
    test_corpus: "tests/mir/fail",
    path_matcher: |path| matches_extension(path, "lisp"),
    expected_path: |path| {
        path.parent().unwrap().join("parse").join(path.file_name().unwrap())
            .with_extension("mir")
    },
    test_runner: |test| {
        let input = std::fs::read(&test.path)?;

        let syntax = match ssa::syntax::Syntax::from_bytes(&input) {
            Err(err) => return Ok(MaybeIgnore::Failed(format!("syntax error: {err}"))),
            Ok(x) => x,
        };

        match ssa::to_mir::Encoder::new().encode(&syntax) {
            Err(err) => Ok(MaybeIgnore::Ran(format!("encoding error: {err}"))),
            Ok(mir) => {
                let mir = ssa::mir::StableDisplayMir::from(mir);
                Ok(MaybeIgnore::Failed(format!("{mir}")))
            }
        }
    }
}

test_source! {
    name: "mir-opts",
    test_corpus: "tests/mir/pass",
    path_matcher: |path| matches_extension(path, "lisp"),
    expected_path: |path| {
        path.parent().unwrap().join("mir-opts").join(path.file_name().unwrap())
            .with_extension("mir")
    },
    test_runner: |test| {
        let input = std::fs::read(&test.path)?;

        let syntax = match ssa::syntax::Syntax::from_bytes(&input) {
            Err(err) => return Ok(MaybeIgnore::Failed(format!("syntax error: {err}"))),
            Ok(x) => x,
        };

        match ssa::to_mir::Encoder::new().encode(&syntax) {
            Err(err) => Ok(MaybeIgnore::Failed(format!("encoding error: {err}"))),
            Ok(mut mir) => {
                ssa::mir_opts::clean_up_jumps(&mut mir);
                let mir = ssa::mir::StableDisplayMir::from(mir);
                Ok(MaybeIgnore::Ran(format!("{mir}")))
            }
        }
    }
}

test_source! {
    name: "ssa",
    test_corpus: "tests/mir/pass",
    path_matcher: |path| matches_extension(path, "lisp"),
    expected_path: |path| {
        path.parent().unwrap().join("ssa").join(path.file_name().unwrap())
            .with_extension("mir")
    },
    test_runner: |test| {
        let input = std::fs::read(&test.path)?;

        let syntax = match ssa::syntax::Syntax::from_bytes(&input) {
            Err(err) => return Ok(MaybeIgnore::Failed(format!("syntax error: {err}"))),
            Ok(x) => x,
        };

        match ssa::to_mir::Encoder::new().encode(&syntax) {
            Err(err) => Ok(MaybeIgnore::Failed(format!("encoding error: {err}"))),
            Ok(mut mir) => {
                ssa::mir_opts::clean_up_jumps(&mut mir);
                let mir = ssa::to_ssa::to_ssa_stable(&mir);
                let mir = ssa::mir::StableDisplayMir::from(mir);
                Ok(MaybeIgnore::Ran(format!("{mir}")))
            }
        }
    }
}

test_source! {
    name: "ssa-opts",
    test_corpus: "tests/mir/pass",
    path_matcher: |path| matches_extension(path, "lisp"),
    expected_path: |path| {
        path.parent().unwrap().join("ssa-opts").join(path.file_name().unwrap())
            .with_extension("mir")
    },
    test_runner: |test| {
        let input = std::fs::read(&test.path)?;

        let syntax = match ssa::syntax::Syntax::from_bytes(&input) {
            Err(err) => return Ok(MaybeIgnore::Failed(format!("syntax error: {err}"))),
            Ok(x) => x,
        };

        match ssa::to_mir::Encoder::new().encode(&syntax) {
            Err(err) => Ok(MaybeIgnore::Failed(format!("encoding error: {err}"))),
            Ok(mut mir) => {
                ssa::mir_opts::clean_up_jumps(&mut mir);
                let mut mir = ssa::to_ssa::to_ssa_stable(&mir);
                ssa::ssa_opts::jump::clean_up_jumps(&mut mir);
                let mir = ssa::mir::StableDisplayMir::from(mir);
                Ok(MaybeIgnore::Ran(format!("{mir}")))
            }
        }
    }
}

fn main() -> Result<(), std::io::Error> {
    mir::harness::run_tests()
}
