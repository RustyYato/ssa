pub mod jump;
pub mod remove_unused;

pub fn opt(mir: &mut crate::mir::Mir) {
    assert!(mir.is_ssa);

    jump::clean_up_jumps(mir);
    remove_unused::remove_unused(mir);
    // jump::clean_up_jumps(mir);
}
