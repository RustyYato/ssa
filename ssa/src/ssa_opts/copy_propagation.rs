use crate::{aliases::*, mir};

pub fn copy_propagation(mir: &mut mir::Mir) {
    assert!(mir.is_ssa);

    let mut regs = RegMap::default();

    for block in mir.blocks.values() {
        for instr in &block.instrs {
            match *instr {
                mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => unreachable!(),
                // even if unused, we must keep all IO
                mir::Instr::ConsoleInput(_) | mir::Instr::ConsolePrint(_) => (),

                mir::Instr::BinOp {
                    op: _,
                    dest: _,
                    left: _,
                    right: _,
                } => (),

                mir::Instr::Store { dest, val } => {
                    regs.insert(dest, val);
                }
            }
        }
    }

    let visit_val = |val: &mut mir::Val| match *val {
        mir::Val::ConstI32(_) | mir::Val::ConstBool(_) | mir::Val::Uninit => (),
        mir::Val::Reg(reg) => {
            if let Some(new_val) = regs.get(&reg) {
                *val = *new_val;
            }
        }
    };

    for block in mir.blocks.values_mut() {
        for instr in &mut block.instrs {
            match instr {
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

        match &mut block.term {
            mir::Terminator::Jump(target) => {
                for arg in &mut target.args {
                    visit_val(arg);
                }
            }
            mir::Terminator::If {
                cond,
                if_true,
                if_false,
            } => {
                visit_val(cond);
                for arg in &mut if_true.args {
                    visit_val(arg);
                }
                for arg in &mut if_false.args {
                    visit_val(arg);
                }
            }
            mir::Terminator::ProgramExit => (),
        }
    }
}
