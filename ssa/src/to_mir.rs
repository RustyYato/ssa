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
    BreakOutOfLoop,
}

impl std::error::Error for EncodingError {}
impl core::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

type Result<T, E = EncodingError> = std::result::Result<T, E>;

#[derive(Debug)]
struct ScopeContext<'a> {
    bb: &'a mut BasicBlockBuilder,
    scopes: ScopeList<'a>,
}

#[derive(Clone, Copy, Debug)]
struct ScopeList<'a> {
    #[cfg(debug_assertions)]
    kind: &'static str,
    id: name_resolver::ScopeRef<'a>,
    after_last_loop: Option<name_resolver::ScopeRef<'a>>,
    current_loop: Option<(
        mir::BasicBlockId,
        mir::BasicBlockId,
        name_resolver::ScopeRef<'a>,
    )>,
    prev: Option<&'a ScopeList<'a>>,
}

impl<'a> ScopeContext<'a> {
    pub fn by_ref(&mut self) -> ScopeContext<'_> {
        ScopeContext {
            bb: self.bb,
            ..*self
        }
    }

    pub fn loop_scope<'b>(
        &'b mut self,
        loop_start: mir::BasicBlockId,
        loop_end: mir::BasicBlockId,
        scope: name_resolver::ScopeRef<'b>,
    ) -> ScopeContext<'_> {
        ScopeContext {
            bb: self.bb,
            scopes: ScopeList {
                #[cfg(debug_assertions)]
                kind: "loop",
                id: scope,
                prev: Some(&self.scopes),
                after_last_loop: None,
                current_loop: Some((loop_start, loop_end, scope)),
            },
        }
    }

    pub fn block_scope<'b>(&'b mut self, scope: name_resolver::ScopeRef<'b>) -> ScopeContext<'_> {
        ScopeContext {
            bb: self.bb,
            scopes: ScopeList {
                #[cfg(debug_assertions)]
                kind: "block",
                id: scope,
                prev: Some(&self.scopes),
                after_last_loop: Some(self.scopes.after_last_loop.unwrap_or(scope)),
                current_loop: self.scopes.current_loop,
            },
        }
    }
}

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

        let scope = self.nr.scope();
        let scopes = ScopeList {
            #[cfg(debug_assertions)]
            kind: "root",
            id: scope.as_ref(),
            prev: None,
            after_last_loop: None,
            current_loop: None,
        };
        self.write_statement(
            syn,
            ScopeContext {
                bb: &mut block,
                scopes,
            },
        )?;
        self.close_scope_unchecked(&mut block, scope);
        self.mir.commit(block, mir::Terminator::ProgramExit);

        Ok(self.mir.finish())
    }

    fn close_scope_unchecked(
        &mut self,
        bb: &mut mir::BasicBlockBuilder,
        scope: name_resolver::ScopeToken,
    ) {
        let vars = self.nr.close_scope_with(scope);

        for var in vars {
            bb.instrs.push(mir::Instr::EndLifetime(var));
        }
    }

    fn close_scope(&mut self, ctx: ScopeContext<'_>, scope: name_resolver::ScopeToken) {
        self.close_scope_unchecked(ctx.bb, scope);
    }

    fn exit_scope(&mut self, ctx: ScopeContext<'_>, scope: name_resolver::ScopeRef<'_>) {
        let vars = self.nr.scope_bindings(ctx.scopes.id);

        for var in vars {
            ctx.bb.instrs.push(mir::Instr::EndLifetime(var));
        }
    }

    fn write_statement(&mut self, syn: &Syntax, mut ctx: ScopeContext<'_>) -> Result<()> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Let) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.define(ident.name, &mut self.regs);
                        ctx.bb.instrs.push(mir::Instr::StartLifetime(reg));
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
                        let val = self.write_expr(value, ctx.by_ref())?;
                        ctx.bb.instrs.push(mir::Instr::Store { dest: reg, val });
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
                    let val = self.write_expr(arg, ctx.by_ref())?;
                    ctx.bb.instrs.push(crate::mir::Instr::ConsolePrint(val));
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForPrint),
                [_, _, ..] => Err(EncodingError::TooManyArgsForPrint),
            },
            Some(keywords::Keyword::Input) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.resolve(ident.name)?;
                        ctx.bb.instrs.push(crate::mir::Instr::ConsoleInput(reg));
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
                let mut block_ctx = ctx.block_scope(scope.as_ref());
                for arg in syn.args.as_slice() {
                    self.write_statement(arg, block_ctx.by_ref())?;
                }
                self.close_scope(ctx, scope);
                Ok(())
            }
            Some(keywords::Keyword::Loop) => {
                let (loop_id, loop_block) = self.mir.new_block();
                let (after_loop_id, after_loop_block) = self.mir.new_block();

                let before_loop = core::mem::replace(ctx.bb, loop_block);
                self.mir.commit(
                    before_loop,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_id)),
                );

                let scope = self.nr.scope();
                let mut loop_ctx = ctx.loop_scope(loop_id, after_loop_id, scope.as_ref());
                for arg in syn.args.as_slice() {
                    self.write_statement(arg, loop_ctx.by_ref())?;
                }

                let loop_block = core::mem::replace(ctx.bb, after_loop_block);

                self.close_scope(ctx, scope);
                self.mir.commit(
                    loop_block,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_id)),
                );

                Ok(())
            }
            Some(keywords::Keyword::Break) => {
                let (_loop_start, loop_end, _loop_scope) = ctx
                    .scopes
                    .current_loop
                    .ok_or(EncodingError::BreakOutOfLoop)?;

                let (_, after_break) = self.mir.new_block();

                let before_break = core::mem::replace(ctx.bb, after_break);

                if let Some(scope) = ctx.scopes.after_last_loop {
                    self.exit_scope(ctx, scope);
                }
                self.mir.commit(
                    before_break,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_end)),
                );

                Ok(())
            }
            Some(keywords::Keyword::Continue) => {
                let (loop_start, _loop_end, _loop_scope) = ctx
                    .scopes
                    .current_loop
                    .ok_or(EncodingError::BreakOutOfLoop)?;

                let (_, after_break) = self.mir.new_block();

                let before_break = core::mem::replace(ctx.bb, after_break);

                if let Some(scope) = ctx.scopes.after_last_loop {
                    self.exit_scope(ctx, scope);
                }
                self.mir.commit(
                    before_break,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_start)),
                );

                Ok(())
            }
            Some(keywords::Keyword::If) => {
                let (cond, if_true, if_false) = match syn.args.as_slice() {
                    [cond, if_true] => (cond, if_true, None),
                    [cond, if_true, if_false] => (cond, if_true, Some(if_false)),
                    _ => todo!(),
                };

                let cond = self.write_expr(cond, ctx.by_ref())?;
                let (exit_id, exit_block) = self.mir.new_block();
                let (if_true_id, if_true_block) = self.mir.new_block();

                let cond_block = core::mem::replace(ctx.bb, if_true_block);

                self.write_statement(if_true, ctx.by_ref())?;

                let if_false_id = if let Some(if_false) = if_false {
                    let (if_false_id, if_false_block) = self.mir.new_block();
                    let branch_block = core::mem::replace(ctx.bb, if_false_block);

                    self.write_statement(if_false, ctx.by_ref())?;

                    self.mir.commit(
                        branch_block,
                        mir::Terminator::Jump(mir::BasicBlockRef::new(exit_id)),
                    );
                    if_false_id
                } else {
                    exit_id
                };

                let branch_block = core::mem::replace(ctx.bb, exit_block);

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

    fn write_expr(&mut self, syn: &Syntax, mut ctx: ScopeContext<'_>) -> Result<mir::Val> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Eq) => {
                let [left, right] = syn.args.as_slice() else {
                    todo!()
                };

                let left = self.write_expr(left, ctx.by_ref())?;
                let right = self.write_expr(right, ctx.by_ref())?;
                let temp = self.regs.create();

                ctx.bb.instrs.push(mir::Instr::CmpEq {
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

                let left = self.write_expr(left, ctx.by_ref())?;
                let right = self.write_expr(right, ctx.by_ref())?;
                let temp = self.regs.create();

                ctx.bb.instrs.push(mir::Instr::Sub {
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
