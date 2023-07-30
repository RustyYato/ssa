mod jump;
mod remove_unused_writes;

pub fn opt(mir: &mut crate::mir::Mir) {
    assert!(mir.is_ssa);

    jump::clean_up_jumps(mir);
    remove_unused_writes::remove_unused_writes(mir);
    jump::clean_up_jumps(mir);
}
