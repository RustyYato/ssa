use hashbrown::{hash_map::Entry, HashMap};

use crate::mir;

enum Access {
    Root,
    Once(mir::BasicBlockId),
    Many,
}

pub fn clean_up_jumps(mir: &mut mir::Mir) {
    // this code doesn't correctly handle block args so it can't handle SSA MIR
    assert!(!mir.is_ssa);

    jump_threading(mir);
    let access_counts = remove_inaccessible_blocks(mir);
    consolidate_single_source_jumps(mir, access_counts);
}

fn jump_threading(mir: &mut mir::Mir) {
    let table = mir.blocks.raw_table_mut();
    let iter = unsafe { table.iter() };

    for bucket in iter {
        let (id, block) = unsafe { bucket.as_mut() };

        let resolve_jump_target = |next: &mut mir::JumpTarget| {
            while next.id != *id {
                let block = &mir.blocks[&next.id];
                match &block.term {
                    mir::Terminator::Jump(next_next) => {
                        if block.instrs.is_empty() {
                            next.clone_from(next_next);
                        } else {
                            break;
                        }
                    }
                    mir::Terminator::If { .. } | mir::Terminator::ProgramExit => break,
                }
            }
        };

        match &mut block.term {
            mir::Terminator::Jump(next) => {
                resolve_jump_target(next);
            }
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                resolve_jump_target(if_true);
                resolve_jump_target(if_false);
            }
            mir::Terminator::ProgramExit => (),
        }
    }
}

fn remove_inaccessible_blocks(mir: &mut mir::Mir) -> HashMap<mir::BasicBlockId, Access> {
    let mut stack = Vec::new();
    stack.push(mir.start);
    let mut accessible = HashMap::new();
    accessible.insert(mir.start, Access::Root);

    let mut access = |id: mir::BasicBlockId, child: mir::BasicBlockId| match accessible.entry(child)
    {
        Entry::Occupied(x) => {
            *x.into_mut() = Access::Many;
            false
        }
        Entry::Vacant(x) => {
            x.insert(Access::Once(id));
            true
        }
    };

    while let Some(bb) = stack.pop() {
        match &mir.blocks[&bb].term {
            mir::Terminator::Jump(next) => {
                if access(bb, next.id) {
                    stack.push(next.id);
                }
            }
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                if access(bb, if_true.id) {
                    stack.push(if_true.id);
                }
                if access(bb, if_false.id) {
                    stack.push(if_false.id);
                }
            }
            mir::Terminator::ProgramExit => (),
        }
    }

    mir.blocks.retain(|id, _block| accessible.contains_key(id));

    accessible
}

fn consolidate_single_source_jumps(
    mir: &mut mir::Mir,
    access_counts: HashMap<mir::BasicBlockId, Access>,
) {
    let mut merged = HashMap::new();

    for (block, access) in access_counts {
        match access {
            Access::Root => (),
            Access::Once(mut parent) => {
                while let Some(next) = merged.get(&parent).copied() {
                    parent = next;
                }

                let [(_, parent), (_, block)] = mir
                    .blocks
                    .get_many_key_value_mut([&parent, &block])
                    .unwrap();

                if let mir::Terminator::Jump(target) = &parent.term {
                    debug_assert_eq!(target.id, block.id);
                    merged.insert(block.id, parent.id);
                    let block = core::mem::replace(block, mir::BasicBlock::invalid());
                    parent.term = block.term;
                    parent.instrs.extend(block.instrs);
                    mir.blocks.remove(&block.id);
                }
            }
            Access::Many => (),
        }
    }
}
