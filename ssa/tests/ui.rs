use std::ffi::OsStr;

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
            let mut parser = ssa::parser::Parser::new(ctx, pool.clear(), &file);
            let file = parser.parse_file();
            *tl_pool = parser.into_pool();
            match serde_gura::to_string(&file) {
                Ok(file) => Ok(Some(Ok(file))),
                Err(err) => Ok(Some(Err(err.to_string()))),
            }
        })
    }
}
