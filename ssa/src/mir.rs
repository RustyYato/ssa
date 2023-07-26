use std::{collections::HashMap, num::NonZeroU32};

#[derive(Debug)]
pub struct Mir {
    blocks: Vec<BasicBlock>,
}

#[derive(Debug)]
pub struct MirBuilder {
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
                args: bb.args,
                instrs: bb.instrs,
                term,
            },
        );
    }

    pub fn finish(mut self) -> Mir {
        let mut blocks = Vec::new();
        for i in 0..self.next_id {
            blocks.push(
                self.blocks
                    .remove(&BasicBlockId(NonZeroU32::new(i + 1).unwrap()))
                    .unwrap(),
            )
        }
        Mir { blocks }
    }
}

impl Default for MirBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlockBuilder {
    id: BasicBlockId,
    pub args: Vec<Reg>,
    pub instrs: Vec<Instr>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlock {
    id: BasicBlockId,
    args: Vec<Reg>,
    instrs: Vec<Instr>,
    term: Terminator,
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
    Store { dest: Reg, val: Val },

    // math ops
    Add { dest: Reg, left: Val, right: Val },
    Mul { dest: Reg, left: Val, right: Val },
    Sub { dest: Reg, left: Val, right: Val },
    Div { dest: Reg, left: Val, right: Val },
    CmpEq { dest: Reg, left: Val, right: Val },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Val {
    ConstI32(i32),
    ConstBool(bool),
    Reg(Reg),
}

pub struct RegAllocator(u32);

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlockRef {
    pub id: BasicBlockId,
    pub args: Vec<Val>,
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

impl BasicBlockRef {
    pub fn new(id: BasicBlockId) -> Self {
        Self {
            id,
            args: Vec::new(),
        }
    }
}
