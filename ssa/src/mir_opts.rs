use hashbrown::{hash_map::Entry, HashMap};

use crate::mir;

enum Access {
    Root,
    Once(mir::BasicBlockId),
    Many,
}

pub fn clean_up_jumps(mir: &mut mir::Mir) {
    assert!(!mir.is_ssa);

    let access_counts = remove_inaccessible_blocks(mir);
    consolidate_single_source_jumps(mir, access_counts);
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
