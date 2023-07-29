use std::num::NonZeroU32;

use hashbrown::HashMap;

#[derive(Debug)]
pub struct Mir {
    pub(crate) is_ssa: bool,
    pub(crate) start: BasicBlockId,
    pub(crate) blocks: HashMap<BasicBlockId, BasicBlock>,
}

#[derive(Debug)]
pub struct StableDisplayMir {
    pub(crate) blocks: Vec<BasicBlock>,
    start: BasicBlockId,
}

impl From<Mir> for StableDisplayMir {
    fn from(mir: Mir) -> Self {
        let mut blocks = Vec::from_iter(mir.blocks.into_values());
        blocks.sort_unstable_by_key(|block| block.id);
        Self {
            start: mir.start,
            blocks,
        }
    }
}

impl core::fmt::Display for Mir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "start {}", self.start)?;
        for (_, block) in &self.blocks {
            write!(f, "{block}")?;
        }
        Ok(())
    }
}

impl core::fmt::Display for StableDisplayMir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "start {}", self.start)?;
        for block in &self.blocks {
            write!(f, "{block}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct MirBuilder {
    next_id: u32,
    blocks: HashMap<BasicBlockId, BasicBlock>,
}

impl MirBuilder {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            blocks: HashMap::default(),
        }
    }

    pub fn new_block(&mut self) -> (BasicBlockId, BasicBlockBuilder) {
        self.next_id += 1;
        let id = NonZeroU32::new(self.next_id).unwrap();

        (
            BasicBlockId(id),
            BasicBlockBuilder {
                id: BasicBlockId(id),
                args: Vec::new(),
                instrs: Vec::new(),
            },
        )
    }

    pub fn commit(&mut self, bb: BasicBlockBuilder, term: Terminator) {
        self.blocks.insert(
            bb.id,
            BasicBlock {
                id: bb.id,
                args: bb.args.into_iter().map(Some).collect(),
                instrs: bb.instrs,
                term,
            },
        );
    }

    pub fn finish(self) -> Mir {
        Mir {
            is_ssa: false,
            start: BasicBlockId(NonZeroU32::new(1).unwrap()),
            blocks: self.blocks,
        }
    }
}

impl Default for MirBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BasicBlockBuilder {
    id: BasicBlockId,
    pub args: Vec<Reg>,
    pub instrs: Vec<Instr>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlock {
    pub(crate) id: BasicBlockId,
    pub(crate) args: Vec<Option<Reg>>,
    pub(crate) instrs: Vec<Instr>,
    pub(crate) term: Terminator,
}
impl BasicBlock {
    pub(crate) fn invalid() -> BasicBlock {
        Self {
            id: BasicBlockId::invalid(),
            args: Vec::new(),
            instrs: Vec::new(),
            term: Terminator::ProgramExit,
        }
    }
}

impl core::fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}{:?}", self.id, self.args)?;
        for instr in &self.instrs {
            writeln!(f, "    {instr}")?
        }
        writeln!(f, "    {}", self.term)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Instr {
    // Lifetime
    StartLifetime(Reg),
    EndLifetime(Reg),

    // Basic IO
    ConsolePrint(Val),
    ConsoleInput(Reg),

    // memory ops
    Store {
        dest: Reg,
        val: Val,
    },

    BinOp {
        op: BinOp,
        dest: Reg,
        left: Val,
        right: Val,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    CmpEq,
}

impl core::fmt::Display for Instr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instr::StartLifetime(reg) => write!(f, "start_lt {reg}"),
            Instr::EndLifetime(reg) => write!(f, "end_lt   {reg}"),
            Instr::ConsolePrint(val) => write!(f, "print {val}"),
            Instr::ConsoleInput(reg) => write!(f, "input {reg}"),
            Instr::Store { dest, val } => write!(f, "{dest} = {val}"),
            Instr::BinOp {
                op: BinOp::CmpEq,
                dest,
                left,
                right,
            } => write!(f, "{dest} = cmp(=, {left}, {right})"),
            Instr::BinOp {
                op,
                dest,
                left,
                right,
            } => {
                let op = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::CmpEq => unreachable!(),
                };
                write!(f, "{dest} = {left} {op} {right}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Val {
    ConstI32(i32),
    ConstBool(bool),
    Reg(Reg),
}

impl core::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val: &dyn core::fmt::Display = match self {
            Val::ConstI32(val) => val,
            Val::ConstBool(val) => val,
            Val::Reg(val) => val,
        };

        val.fmt(f)
    }
}

pub(crate) struct RegAllocator(u32);

impl RegAllocator {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn create(&mut self) -> Reg {
        self.0 += 1;
        Reg(NonZeroU32::new(self.0).unwrap())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg(NonZeroU32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlockId(NonZeroU32);

impl BasicBlockId {
    pub fn to_usize(self) -> usize {
        self.0.get().wrapping_sub(1) as usize
    }

    pub(crate) fn invalid() -> Self {
        BasicBlockId(NonZeroU32::new(u32::MAX).unwrap())
    }

    pub(crate) fn normal_start() -> Self {
        BasicBlockId(NonZeroU32::new(1).unwrap())
    }
}

impl core::fmt::Debug for Reg {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{}", self.0)
    }
}

impl core::fmt::Display for Reg {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl core::fmt::Debug for BasicBlockId {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl core::fmt::Display for BasicBlockId {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlockRef {
    pub id: BasicBlockId,
    pub args: Vec<Val>,
}

impl Clone for BasicBlockRef {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            args: self.args.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self { id, args } = self;

        *id = source.id;
        args.clone_from(&source.args);
    }
}

impl core::fmt::Display for BasicBlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{:?}", self.id, self.args)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Terminator {
    Jump(BasicBlockRef),
    If {
        cond: Val,
        if_true: BasicBlockRef,
        if_false: BasicBlockRef,
    },
    ProgramExit,
}

impl core::fmt::Display for Terminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Terminator::Jump(next) => write!(f, "jmp {next}"),
            Terminator::If {
                cond,
                if_true,
                if_false,
            } => write!(f, "if {cond} {if_true} {if_false}"),
            Terminator::ProgramExit => write!(f, "exit"),
        }
    }
}

impl BasicBlockRef {
    pub fn new(id: BasicBlockId) -> Self {
        Self {
            id,
            args: Vec::new(),
        }
    }
}
