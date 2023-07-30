use hashbrown::{hash_map::Entry, HashMap};

use crate::mir;

enum Access {
    Root,
    Once(mir::BasicBlockId),
    Many,
}

pub fn clean_up_jumps(mir: &mut mir::Mir) {
    assert!(mir.is_ssa);

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
                        if next_next.id != next.id && block.instrs.is_empty() {
                            let mut next_next = next_next.clone();
                            write_args_to_jump_target(&mut next_next, &block.args, next);
                            *next = next_next;
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
                    write_args(block, target);

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

fn write_args(block: &mut mir::BasicBlock, target: &mir::JumpTarget) {
    debug_assert_eq!(target.id, block.id);
    assert_eq!(target.args.len(), block.args.len());

    if target.args.is_empty() {
        return;
    }

    for instr in &mut block.instrs {
        match instr {
            mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => unreachable!(),
            mir::Instr::ConsolePrint(val) => write_args_to_val(val, &block.args, target),
            mir::Instr::ConsoleInput(_) => todo!(),
            mir::Instr::Store { dest: _, val } => {
                write_args_to_val(val, &block.args, target);
            }
            mir::Instr::BinOp {
                op: _,
                dest: _,
                left,
                right,
            } => {
                write_args_to_val(left, &block.args, target);
                write_args_to_val(right, &block.args, target);
            }
        }
    }

    match &mut block.term {
        mir::Terminator::Jump(next) => write_args_to_jump_target(next, &block.args, target),
        mir::Terminator::If {
            cond,
            if_true,
            if_false,
        } => {
            write_args_to_val(cond, &block.args, target);
            write_args_to_jump_target(if_true, &block.args, target);
            write_args_to_jump_target(if_false, &block.args, target);
        }
        mir::Terminator::ProgramExit => (),
    }
}

fn write_args_to_jump_target(
    dest_target: &mut mir::JumpTarget,
    args: &[Option<mir::Reg>],
    source_target: &mir::JumpTarget,
) {
    for val in &mut dest_target.args {
        write_args_to_val(val, args, source_target);
    }
}

fn write_args_to_val(
    val: &mut mir::Val,
    args: &[Option<mir::Reg>],
    source_target: &mir::JumpTarget,
) {
    let reg = match *val {
        mir::Val::ConstI32(_) | mir::Val::ConstBool(_) | mir::Val::Uninit => return,
        mir::Val::Reg(reg) => reg,
    };

    for (name, new_val) in args.iter().zip(&source_target.args) {
        if Some(reg) == *name {
            *val = *new_val;
            break;
        }
    }
}
