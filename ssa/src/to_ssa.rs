use crate::{
    mir::{self, BasicBlockBuilder, MirBuilder},
    syntax::Syntax,
};

mod keywords;
mod name_resolver;

pub struct Encoder {
    keywords: keywords::Keywords,
    mir: MirBuilder,
    nr: name_resolver::NameResolver,
    regs: mir::RegAllocator,
}

#[derive(Debug)]
pub enum EncodingError {
    MissingArgsForPrint,
    TooManyArgsForPrint,
    MissingArgsForInput,
    TooManyArgsForInput,
    UnresolvedIdent(istr::IStr),
    TooManyArgsForSet,
    MissingArgsForSet,
    MissingArgsForLet,
    TooManyArgsForLet,
}

impl std::error::Error for EncodingError {}
impl core::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

type Result<T, E = EncodingError> = std::result::Result<T, E>;

impl Encoder {
    pub fn new() -> Self {
        Self {
            keywords: keywords::Keywords::init(),
            mir: MirBuilder::new(),
            nr: name_resolver::NameResolver::new(),
            regs: mir::RegAllocator::new(),
        }
    }

    pub fn encode(mut self, syn: &Syntax) -> Result<mir::Mir> {
        let (_id, mut block) = self.mir.new_block();

        self.write_statement(syn, &mut block)?;
        self.mir.commit(block, mir::Terminator::ProgramExit);

        Ok(self.mir.finish())
    }

    fn write_statement(&mut self, syn: &Syntax, bb: &mut BasicBlockBuilder) -> Result<()> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Let) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.define(ident.name, &mut self.regs);
                        // bb.instrs.push(mir::Instr::StartLifetime(reg));
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [name, val] => {
                    todo!();
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForLet),
                [_, _, _, ..] => Err(EncodingError::TooManyArgsForLet),
            },
            Some(keywords::Keyword::Set) => match syn.args.as_slice() {
                [ident, value] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.resolve(ident.name)?;
                        let val = self.write_expr(value, bb)?;
                        bb.instrs.push(mir::Instr::Store { dest: reg, val });
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForSet),
                [_, ..] => Err(EncodingError::TooManyArgsForSet),
            },

            Some(keywords::Keyword::Print) => match syn.args.as_slice() {
                [arg] => {
                    let val = self.write_expr(arg, bb)?;
                    bb.instrs.push(crate::mir::Instr::ConsolePrint(val));
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForPrint),
                [_, _, ..] => Err(EncodingError::TooManyArgsForPrint),
            },
            Some(keywords::Keyword::Input) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.resolve(ident.name)?;
                        bb.instrs.push(crate::mir::Instr::ConsoleInput(reg));
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForPrint),
                [_, _, ..] => Err(EncodingError::TooManyArgsForPrint),
            },
            Some(keywords::Keyword::Block) => {
                let scope = self.nr.scope();
                for arg in syn.args.as_slice() {
                    self.write_statement(arg, bb)?;
                }
                self.nr.close_scope(scope);
                // for reg in regs {
                //     bb.instrs.push(mir::Instr::EndLifetime(reg));
                // }
                Ok(())
            }
            Some(keywords::Keyword::If) => {
                let (cond, if_true, if_false) = match syn.args.as_slice() {
                    [cond, if_true] => (cond, if_true, None),
                    [cond, if_true, if_false] => (cond, if_true, Some(if_false)),
                    _ => todo!(),
                };

                let cond = self.write_expr(cond, bb)?;
                let (exit_id, exit_block) = self.mir.new_block();
                let (if_true_id, if_true_block) = self.mir.new_block();

                let cond_block = core::mem::replace(bb, if_true_block);

                self.write_statement(if_true, bb)?;

                let if_false_id = if let Some(if_false) = if_false {
                    let (if_false_id, if_false_block) = self.mir.new_block();
                    let branch_block = core::mem::replace(bb, if_false_block);

                    self.write_statement(if_false, bb)?;

                    self.mir.commit(
                        branch_block,
                        mir::Terminator::Jump(mir::BasicBlockRef::new(exit_id)),
                    );
                    if_false_id
                } else {
                    exit_id
                };

                let branch_block = core::mem::replace(bb, exit_block);

                self.mir.commit(
                    branch_block,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(exit_id)),
                );

                self.mir.commit(
                    cond_block,
                    mir::Terminator::If {
                        cond,
                        if_true: mir::BasicBlockRef::new(if_true_id),
                        if_false: mir::BasicBlockRef::new(if_false_id),
                    },
                );

                Ok(())
            }
            _ => todo!(),
        }
    }

    fn write_expr(&mut self, syn: &Syntax, bb: &mut BasicBlockBuilder) -> Result<mir::Val> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Eq) => {
                let [left, right] = syn.args.as_slice() else {
                    todo!()
                };

                let left = self.write_expr(left, bb)?;
                let right = self.write_expr(right, bb)?;
                let temp = self.regs.create();

                bb.instrs.push(mir::Instr::CmpEq {
                    dest: temp,
                    left,
                    right,
                });

                Ok(mir::Val::Reg(temp))
            }
            Some(keywords::Keyword::Sub) => {
                let [left, right] = syn.args.as_slice() else {
                    todo!()
                };

                let left = self.write_expr(left, bb)?;
                let right = self.write_expr(right, bb)?;
                let temp = self.regs.create();

                bb.instrs.push(mir::Instr::Sub {
                    dest: temp,
                    left,
                    right,
                });

                Ok(mir::Val::Reg(temp))
            }
            Some(_) => {
                todo!()
            }
            None => {
                if syn.args.is_empty() {
                    if syn.name.bytes().all(|b| b.is_ascii_digit()) {
                        Ok(mir::Val::ConstI32(syn.name.parse().unwrap()))
                    } else {
                        Ok(mir::Val::Reg(self.nr.resolve(syn.name)?))
                    }
                } else {
                    todo!()
                }
            }
        }
    }
}
