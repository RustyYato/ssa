use crate::{aliases::*, mir};

pub fn remove_unused(mir: &mut mir::Mir) {
    assert!(mir.is_ssa);

    let mut regs = RegSet::default();

    let mut visit_val = |val: mir::Val| match val {
        mir::Val::ConstI32(_) | mir::Val::ConstBool(_) | mir::Val::Uninit => (),
        mir::Val::Reg(reg) => {
            regs.insert(reg);
        }
    };

    for block in mir.blocks.values() {
        for instr in &block.instrs {
            match *instr {
                mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => unreachable!(),
                mir::Instr::ConsoleInput(_) => (),
                mir::Instr::ConsolePrint(val) | mir::Instr::Store { dest: _, val } => {
                    visit_val(val)
                }
                mir::Instr::BinOp {
                    op: _,
                    dest: _,
                    left,
                    right,
                } => {
                    visit_val(left);
                    visit_val(right);
                }
            }
        }

        match &block.term {
            mir::Terminator::Jump(target) => {
                for arg in &target.args {
                    visit_val(*arg);
                }
            }
            mir::Terminator::If {
                cond,
                if_true,
                if_false,
            } => {
                visit_val(*cond);
                for arg in &if_true.args {
                    visit_val(*arg);
                }
                for arg in &if_false.args {
                    visit_val(*arg);
                }
            }
            mir::Terminator::ProgramExit => (),
        }
    }

    for block in mir.blocks.values_mut() {
        block.instrs.retain(|instr| match *instr {
            mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => unreachable!(),
            // even if unused, we must keep all IO
            mir::Instr::ConsoleInput(_) | mir::Instr::ConsolePrint(_) => true,

            mir::Instr::Store { dest, val: _ }
            | mir::Instr::BinOp {
                op: _,
                dest,
                left: _,
                right: _,
            } => regs.contains(&dest),
        });
    }
}
