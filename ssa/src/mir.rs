use std::{
    collections::{BTreeMap, HashSet},
    num::NonZeroU32,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Register {
    value: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasicBlockId {
    value: NonZeroU32,
}

#[derive(Debug, Default)]
pub struct Program {
    predecessors: BTreeMap<BasicBlockId, Vec<BasicBlockId>>,
    basic_blocks: BTreeMap<BasicBlockId, BasicBlock>,
    unfilled_blocks: HashSet<BasicBlockId>,
    next_bb: u32,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn starr(&self) -> Option<&BasicBlock> {
        self.basic_blocks.first_key_value().map(|(_, v)| v)
    }

    pub fn blocks(&self) -> impl Iterator<Item = &BasicBlock> + Clone {
        self.basic_blocks.values()
    }

    pub fn blocks_mut(&mut self) -> impl Iterator<Item = &mut BasicBlock> {
        self.basic_blocks.values_mut()
    }

    pub fn alloc_block(&mut self) -> BasicBlockBuilder {
        const ONE: NonZeroU32 = NonZeroU32::MIN;
        const _: () = assert!(ONE.get() == 1);

        let bb = ONE
            .checked_add(self.next_bb)
            .expect("Tried to create too many basic blocks");

        BasicBlockBuilder {
            data: BasicBlock {
                id: BasicBlockId { value: bb },
                term: Terminator::Return,
                pred: Vec::new(),
                instrs: Vec::new(),
            },
        }
    }

    pub fn push(&mut self, block: BasicBlockBuilder) {
        assert!(self.unfilled_blocks.remove(&block.data.id));
        debug_assert!(self
            .basic_blocks
            .last_key_value()
            .map_or(true, |(key, _)| *key < block.id()));

        self.basic_blocks.insert(block.data.id, block.data);
    }

    pub fn validate_reachable(&self) {
        for x in self.basic_blocks.values() {
            match x.term {
                Terminator::Return => (),
                Terminator::Jmp { target } => {
                    assert!(self.basic_blocks.contains_key(&target));
                }
                Terminator::If {
                    condition: _,
                    target_a,
                    target_b,
                } => {
                    assert!(self.basic_blocks.contains_key(&target_a));
                    assert!(self.basic_blocks.contains_key(&target_b));
                }
            }
        }
    }

    pub fn set_pred(&mut self) {
        self.validate_reachable();

        // trim away any removed keys, and clear all other vectors since they are going to be
        // computed now
        self.predecessors.retain(|id, x| {
            x.clear();
            self.basic_blocks.contains_key(id)
        });

        let mut add_pred = |pred: BasicBlockId, bb: BasicBlockId| {
            self.predecessors.entry(bb).or_default().push(pred);
        };

        for x in self.basic_blocks.values_mut() {
            match x.term {
                Terminator::Return => todo!(),
                Terminator::Jmp { target } => add_pred(x.id, target),
                Terminator::If {
                    condition: _,
                    target_a,
                    target_b,
                } => {
                    add_pred(x.id, target_a);
                    add_pred(x.id, target_b);
                }
            }
        }
    }

    pub fn remove_all(&mut self, iter: impl IntoIterator<Item = BasicBlockId>) {
        for id in iter {
            self.basic_blocks.remove(&id);
        }
    }
}

impl core::ops::Index<BasicBlockId> for Program {
    type Output = BasicBlock;

    fn index(&self, index: BasicBlockId) -> &Self::Output {
        &self.basic_blocks[&index]
    }
}

impl core::ops::IndexMut<BasicBlockId> for Program {
    fn index_mut(&mut self, index: BasicBlockId) -> &mut Self::Output {
        self.basic_blocks
            .get_mut(&index)
            .expect("no entry found for key")
    }
}

pub struct BasicBlockBuilder {
    data: BasicBlock,
}

impl BasicBlockBuilder {
    pub fn id(&self) -> BasicBlockId {
        self.data.id
    }

    pub fn set_term(&mut self, term: Terminator) {
        self.data.term = term;
    }

    pub fn push(&mut self, instr: Instr) {
        self.data.instrs.push(instr);
    }
}

#[derive(Debug)]
pub struct BasicBlock {
    id: BasicBlockId,
    pub term: Terminator,
    pub pred: Vec<BasicBlockId>,
    pub instrs: Vec<Instr>,
}

impl BasicBlock {
    pub fn id(&self) -> BasicBlockId {
        self.id
    }
}

impl Eq for BasicBlock {}
impl PartialEq for BasicBlock {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd for BasicBlock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BasicBlock {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Instr {
    // Move the contents of source into value
    Move {
        dest: Register,
        source: Value,
    },
    // do the given binary operation, with the two operands
    BinaryOp {
        op: BinaryOp,
        dest: Register,
        left: Value,
        right: Value,
    },
    // do the given unary operation
    UnaryOp {
        op: UnaryOp,
        dest: Register,
        source: Value,
    },
    // read an int from the user and store it in dest
    Read {
        dest: Register,
    },
    // print an int or bool to the user
    Print {
        source: Value,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    // int op int => int
    Add,
    Sub,
    Mul,
    Div,

    // int op int = bool
    // bool op bool => bool
    CmpEq,
    CmpNe,
    CmpGt,
    CmpLt,
    CmpGe,
    CmpLe,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    // bool => int
    IntFromBool,
    // bool => bool
    // int => int
    Not,
    // int => int
    Neg,
}

#[derive(Debug, Clone, Copy)]
pub enum Terminator {
    // Return from function
    Return,
    // Jump to given block
    Jmp {
        target: BasicBlockId,
    },
    // Jump to target_a if condition is true
    // Jump to target_b if condition is false
    If {
        condition: Value,
        target_a: BasicBlockId,
        target_b: BasicBlockId,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Value {
    Register(Register),
    ConstBool(bool),
    ConstI32(i32),
}
