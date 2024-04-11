use std::ffi::OsStr;

use ssa::parser::{HadErrors, ParseError};

fn main() -> ! {
    test_harness::main()
}

test_harness::test_corpus! {
    test_harness::TestDirectory::new("ui/parse", "tests/ui/parse", UiTestCorpus)
}

struct UiTestCorpus;

impl test_harness::PathTestRunner for UiTestCorpus {
    fn is_test_path(&self, path: &std::path::Path) -> bool {
        path.extension().is_some_and(|ext| ext == OsStr::new("ssa"))
    }

    fn expected_output_path(&self, path: &std::path::Path) -> std::path::PathBuf {
        let mut path = test_harness::expected_in_directory(path, "expected");
        path.set_extension("ssa.expected");
        path
    }

    fn run_test(&self, path: &std::path::Path) -> std::io::Result<Option<Result<String, String>>> {
        use ssa::parser::{AstContext, ObjectPools};
        std::thread_local! {
            static ALLOC: core::cell::RefCell<(AstContext, ObjectPools<'static>)> = Default::default();
        }

        ALLOC.with(|ctx| {
            let (ctx, tl_pool) = &mut *ctx.borrow_mut();
            ctx.reset();
            let pool = core::mem::take(tl_pool);

            let file = std::fs::read(path)?;
            let mut errors = TestError::default().had_errors();
            let mut parser = ssa::parser::Parser::new(ctx, pool.clear(), &mut errors, &file);
            let file = parser.parse_file();
            *tl_pool = parser.into_pool();
            if errors.had_errors {
                Ok(Some(Ok(errors.errors.errors.join("\n"))))
            } else {
                match serde_gura::to_string(&file) {
                    Ok(file) => Ok(Some(Ok(file))),
                    Err(err) => Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        err.to_string(),
                    )),
                }
            }
        })
    }
}

#[derive(Default, serde::Serialize)]
#[serde(transparent)]
pub struct TestError {
    errors: Vec<String>,
}

impl<'text> ssa::parser::ParseError<'text> for TestError {
    fn expected_item(&mut self, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!("expected item {found:#?}"))
    }

    fn expected_ident(&mut self, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!("expected ident {found:#?}"))
    }

    fn expected_expr(&mut self, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!("expected expr {found:#?}"))
    }

    fn expected_type(&mut self, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!("expected type {found:#?}"))
    }

    fn too_many_bits(&mut self, ty: char, bits: &str, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!(
            "too many bits ({bits}) for {} {found:#?}",
            match ty {
                'i' => "signed integer",
                'u' => "unsigned integer",
                'f' => "float",
                _ => "unknown",
            }
        ))
    }

    fn zero_bit_integer(&mut self, is_signed: bool, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!(
            "zero bits {}signed integer {found:#?}",
            match is_signed {
                true => "",
                false => "un",
            }
        ))
    }

    fn unsupported_float_bits(&mut self, bits: u16, found: &[ssa::parser::Token<'text>]) {
        self.errors.push(format!(
            "unsupported number of btis for float {bits} {found:#?}"
        ))
    }
}
