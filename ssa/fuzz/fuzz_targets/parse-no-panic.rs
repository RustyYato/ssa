#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|text: &[u8]| {
    let ctx = ssa::parser::AstContext::default();
    ssa::parser::Parser::new(
        &ctx,
        ssa::parser::ObjectPools::default(),
        &mut ssa::parser::HadErrors::new(),
        text,
    )
    .parse_file();
});

#[no_mangle]
pub extern "C-unwind" fn __lexer_error_unreachable_default() -> ! {
    panic!("unreachable")
}
