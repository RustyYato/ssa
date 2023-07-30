use std::cell::Cell;

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
    scopes: ScopeInfo<'a>,
}

#[derive(Debug)]
struct LoopInfo {
    break_used: Cell<bool>,
    continue_used: Cell<bool>,
}

#[derive(Clone, Copy, Debug)]
struct ScopeInfo<'a> {
    #[cfg(debug_assertions)]
    _kind: &'static str,
    id: name_resolver::ScopeRef<'a>,
    after_last_loop: Option<name_resolver::ScopeRef<'a>>,
    current_loop: Option<(mir::BasicBlockId, mir::BasicBlockId, &'a LoopInfo)>,
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
        loop_exit: mir::BasicBlockId,
        scope: name_resolver::ScopeRef<'b>,
        loop_info: &'b LoopInfo,
    ) -> ScopeContext<'_> {
        ScopeContext {
            bb: self.bb,
            scopes: ScopeInfo {
                #[cfg(debug_assertions)]
                _kind: "loop",
                id: scope,
                // prev: Some(&self.scopes),
                after_last_loop: None,
                current_loop: Some((loop_start, loop_exit, loop_info)),
            },
        }
    }

    pub fn block_scope<'b>(&'b mut self, scope: name_resolver::ScopeRef<'b>) -> ScopeContext<'_> {
        ScopeContext {
            bb: self.bb,
            scopes: ScopeInfo {
                #[cfg(debug_assertions)]
                _kind: "block",
                id: scope,
                // prev: Some(&self.scopes),
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
        let scopes = ScopeInfo {
            #[cfg(debug_assertions)]
            _kind: "root",
            id: scope.as_ref(),
            // prev: None,
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

    fn exit_scope_unchecked(
        &mut self,
        bb: &mut BasicBlockBuilder,
        scope: name_resolver::ScopeRef<'_>,
    ) {
        let vars = self.nr.scope_bindings(scope);

        for var in vars {
            bb.instrs.push(mir::Instr::EndLifetime(var));
        }
    }

    fn exit_scope(&mut self, ctx: ScopeContext<'_>, scope: name_resolver::ScopeRef<'_>) {
        self.exit_scope_unchecked(ctx.bb, scope)
    }

    fn block_scope<R>(
        &mut self,
        mut ctx: ScopeContext<'_>,
        f: impl FnOnce(&mut Self, ScopeContext<'_>) -> Result<R>,
    ) -> Result<R> {
        let scope = self.nr.scope();
        let sub_ctx = ctx.block_scope(scope.as_ref());
        let output = f(self, sub_ctx)?;
        self.close_scope(ctx, scope);
        Ok(output)
    }

    fn loop_scope<R>(
        &mut self,
        mut ctx: ScopeContext<'_>,
        loop_start: mir::BasicBlockId,
        loop_exit: mir::BasicBlockId,
        f: impl FnOnce(&mut Self, ScopeContext<'_>, &LoopInfo) -> Result<R>,
    ) -> Result<R> {
        let loop_info = LoopInfo {
            break_used: Cell::new(false),
            continue_used: Cell::new(false),
        };
        let scope = self.nr.scope();
        let sub_ctx = ctx.loop_scope(loop_start, loop_exit, scope.as_ref(), &loop_info);
        let output = f(self, sub_ctx, &loop_info)?;
        self.close_scope(ctx, scope);
        Ok(output)
    }

    fn write_statement(&mut self, syn: &Syntax, mut ctx: ScopeContext<'_>) -> Result<()> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Let) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.define(ident.name, &mut self.regs);
                        ctx.bb.instrs.push(mir::Instr::StartLifetime(reg));
                        ctx.bb.instrs.push(mir::Instr::WriteUninit(reg));
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [ident, val] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.define(ident.name, &mut self.regs);
                        ctx.bb.instrs.push(mir::Instr::StartLifetime(reg));
                        let val = self.write_expr(val, ctx.by_ref())?;
                        ctx.bb.instrs.push(mir::Instr::Store { dest: reg, val });
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForLet),
                [_, _, ..] => Err(EncodingError::TooManyArgsForLet),
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
            Some(keywords::Keyword::Block) => self.block_scope(
                ctx.by_ref(),
                |this: &mut Self, mut ctx: ScopeContext<'_>| {
                    for arg in syn.args.as_slice() {
                        this.write_statement(arg, ctx.by_ref())?;
                    }

                    Ok(())
                },
            ),
            Some(keywords::Keyword::Loop) => {
                let (loop_body, loop_body_block) = self.mir.new_block();
                let (loop_start, loop_restart_block) = self.mir.new_block();
                let (loop_exit, mut loop_exit_block) = self.mir.new_block();

                let before_loop = core::mem::replace(ctx.bb, loop_body_block);
                self.mir.commit(
                    before_loop,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_body)),
                );

                let loop_block = self.loop_scope(
                    ctx.by_ref(),
                    loop_start,
                    loop_exit,
                    |this, mut ctx, loop_info| {
                        for arg in syn.args.as_slice() {
                            this.write_statement(arg, ctx.by_ref())?;
                        }

                        if loop_info.break_used.get() {
                            this.exit_scope_unchecked(&mut loop_exit_block, ctx.scopes.id);
                        }

                        Ok(core::mem::replace(ctx.bb, loop_restart_block))
                    },
                )?;

                let restart_loop_block = core::mem::replace(ctx.bb, loop_exit_block);

                self.mir.commit(
                    loop_block,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_start)),
                );

                self.mir.commit(
                    restart_loop_block,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_body)),
                );

                Ok(())
            }
            Some(keywords::Keyword::Break) => {
                let (_loop_start, loop_end, loop_info) = ctx
                    .scopes
                    .current_loop
                    .ok_or(EncodingError::BreakOutOfLoop)?;

                loop_info.break_used.set(true);

                let (_, after_break) = self.mir.new_block();

                if let Some(scope) = ctx.scopes.after_last_loop {
                    self.exit_scope(ctx.by_ref(), scope);
                }

                let before_break = core::mem::replace(ctx.bb, after_break);

                self.mir.commit(
                    before_break,
                    mir::Terminator::Jump(mir::BasicBlockRef::new(loop_end)),
                );

                Ok(())
            }
            Some(keywords::Keyword::Continue) => {
                let (loop_start, _loop_end, loop_info) = ctx
                    .scopes
                    .current_loop
                    .ok_or(EncodingError::BreakOutOfLoop)?;

                loop_info.continue_used.set(true);

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

                self.block_scope(ctx.by_ref(), |this: &mut Self, ctx: ScopeContext<'_>| {
                    this.write_statement(if_true, ctx)
                })?;

                let if_false_id = if let Some(if_false) = if_false {
                    let (if_false_id, if_false_block) = self.mir.new_block();
                    let branch_block = core::mem::replace(ctx.bb, if_false_block);

                    self.block_scope(ctx.by_ref(), |this: &mut Self, ctx: ScopeContext<'_>| {
                        this.write_statement(if_false, ctx)
                    })?;

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

    fn binary_op(
        &mut self,
        syn: &Syntax,
        mut ctx: ScopeContext<'_>,
        f: impl FnOnce(mir::Reg, mir::Val, mir::Val) -> mir::Instr,
    ) -> Result<mir::Val> {
        let [left, right] = syn.args.as_slice() else {
            todo!()
        };

        let left = self.write_expr(left, ctx.by_ref())?;
        let right = self.write_expr(right, ctx.by_ref())?;
        let temp = self.regs.create();

        ctx.bb.instrs.push(f(temp, left, right));

        Ok(mir::Val::Reg(temp))
    }

    fn write_expr(&mut self, syn: &Syntax, ctx: ScopeContext<'_>) -> Result<mir::Val> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Eq) => {
                self.binary_op(syn, ctx, |dest, left, right| mir::Instr::BinOp {
                    op: mir::BinOp::CmpEq,
                    dest,
                    left,
                    right,
                })
            }
            Some(keywords::Keyword::Add) => {
                self.binary_op(syn, ctx, |dest, left, right| mir::Instr::BinOp {
                    op: mir::BinOp::Add,
                    dest,
                    left,
                    right,
                })
            }
            Some(keywords::Keyword::Sub) => {
                self.binary_op(syn, ctx, |dest, left, right| mir::Instr::BinOp {
                    op: mir::BinOp::Sub,
                    dest,
                    left,
                    right,
                })
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

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}
