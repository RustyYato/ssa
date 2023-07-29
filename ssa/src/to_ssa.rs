use std::{collections::VecDeque, fmt::Debug};

use hashbrown::{HashMap, HashSet};

use crate::mir;

use petgraph::{
    adj, data,
    visit::{self, Visitable},
};

#[derive(Clone, Copy)]
struct MirGraph<'a> {
    mir: &'a mir::Mir,
    max: mir::BasicBlockId,
    predecessors: &'a HashMap<mir::BasicBlockId, Vec<mir::BasicBlockId>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeId(mir::BasicBlockId);

impl Default for NodeId {
    fn default() -> Self {
        Self(mir::BasicBlockId::invalid())
    }
}

impl<'a> visit::GraphBase for MirGraph<'a> {
    #[doc = " edge identifier"]
    type EdgeId = (mir::BasicBlockId, mir::BasicBlockId);

    #[doc = " node identifier"]
    type NodeId = NodeId;
}

impl<'a> visit::Data for MirGraph<'a> {
    type NodeWeight = ();
    type EdgeWeight = ();
}

unsafe impl adj::IndexType for NodeId {
    fn new(_x: usize) -> Self {
        unreachable!("dominators doesn't need to create a NodeId, only access it")
    }

    fn index(&self) -> usize {
        self.0.to_usize()
    }

    fn max() -> Self {
        unreachable!("dominators doesn't need to create a NodeId, only access it")
    }
}

impl<'a> data::DataMap for MirGraph<'a> {
    fn node_weight(&self, id: Self::NodeId) -> Option<&Self::NodeWeight> {
        self.mir.blocks.get(&id.0).map(|_| &())
    }

    fn edge_weight(&self, (source, dest): Self::EdgeId) -> Option<&Self::EdgeWeight> {
        match &self.mir.blocks.get(&source)?.term {
            mir::Terminator::Jump(next) => {
                if next.id == dest {
                    return Some(&());
                }
            }
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                if if_true.id == dest {
                    return Some(&());
                }

                if if_false.id == dest {
                    return Some(&());
                }
            }
            mir::Terminator::ProgramExit => (),
        }

        None
    }
}

impl<'a> visit::IntoNodeIdentifiers for MirGraph<'a> {
    type NodeIdentifiers = std::iter::Map<
        std::iter::Copied<hashbrown::hash_map::Keys<'a, mir::BasicBlockId, mir::BasicBlock>>,
        fn(mir::BasicBlockId) -> NodeId,
    >;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.mir.blocks.keys().copied().map(NodeId)
    }
}

impl visit::GraphRef for MirGraph<'_> {}

impl visit::IntoNeighbors for MirGraph<'_> {
    type Neighbors = std::array::IntoIter<NodeId, 2>;

    #[doc = " Return an iterator of the neighbors of node `a`."]
    fn neighbors(self, a: Self::NodeId) -> Self::Neighbors {
        match &self.mir.blocks[&a.0].term {
            mir::Terminator::Jump(next) => {
                let mut iter = [NodeId(self.mir.start), NodeId(next.id)].into_iter();
                iter.next();
                iter
            }
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => [NodeId(if_true.id), NodeId(if_false.id)].into_iter(),
            mir::Terminator::ProgramExit => {
                let mut iter = [NodeId(self.mir.start), NodeId(self.mir.start)].into_iter();
                iter.next();
                iter.next();
                iter
            }
        }
    }
}

impl visit::Visitable for MirGraph<'_> {
    #[doc = " The associated map type"]
    type Map = fixedbitset::FixedBitSet;

    #[doc = " Create a new visitor map"]
    fn visit_map(&self) -> Self::Map {
        let mut map = fixedbitset::FixedBitSet::new();
        self.reset_map(&mut map);
        map
    }

    #[doc = " Reset the visitor map (and resize to new size of graph if needed)"]
    fn reset_map(&self, map: &mut Self::Map) {
        map.clear();
        map.grow(self.max.to_usize() + 1);
    }
}

struct Nodes<'a> {
    slice: std::slice::Iter<'a, mir::BasicBlockId>,
}

impl Iterator for Nodes<'_> {
    type Item = NodeId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next().copied().map(NodeId)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.slice.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.slice.nth(n).copied().map(NodeId)
    }
}

impl<'a> visit::IntoNeighborsDirected for MirGraph<'a> {
    type NeighborsDirected = either::Either<<Self as visit::IntoNeighbors>::Neighbors, Nodes<'a>>;

    fn neighbors_directed(
        self,
        n: Self::NodeId,
        d: petgraph::Direction,
    ) -> Self::NeighborsDirected {
        match d {
            petgraph::Direction::Outgoing => {
                either::Either::Left(<Self as visit::IntoNeighbors>::neighbors(self, n))
            }
            petgraph::Direction::Incoming => either::Either::Right(Nodes {
                slice: self.predecessors[&n.0].iter(),
            }),
        }
    }
}

type Reg2Reg = HashMap<mir::Reg, mir::Reg>;
type BlockMap<T> = HashMap<mir::BasicBlockId, T>;

struct SsaBuilder<'a> {
    final_name_assginments: &'a BlockMap<(mir::RegAllocator, Reg2Reg)>,
    dominators: &'a petgraph::algo::dominators::Dominators<NodeId>,
    block_args: &'a BlockMap<Vec<(mir::Reg, mir::Reg)>>,

    builder: mir::MirBuilder,

    block_mapping: BlockMap<mir::BasicBlockId>,

    queue: VecDeque<Action<'a>>,
    visited: fixedbitset::FixedBitSet,
}

impl<'a> SsaBuilder<'a> {
    fn visit(&mut self, block_id: mir::BasicBlockId) {
        if !self.visited.contains(block_id.to_usize()) {
            self.visited.insert(block_id.to_usize());
            self.queue.push_back(Action::MakeBlockSsa(block_id));
        }
    }

    fn resolve(&self, nr: &Reg2Reg, block_id: mir::BasicBlockId, val: mir::Val) -> mir::Val {
        let reg = match val {
            mir::Val::ConstI32(_) | mir::Val::ConstBool(_) => return val,
            mir::Val::Reg(reg) => reg,
        };

        mir::Val::Reg(self.resolve_register(nr, block_id, reg))
    }

    fn resolve_register(
        &self,
        nr: &Reg2Reg,
        block_id: mir::BasicBlockId,
        reg: mir::Reg,
    ) -> mir::Reg {
        if let Some(&reg) = nr.get(&reg) {
            return reg;
        }

        self.dominators
            .strict_dominators(NodeId(block_id))
            .unwrap()
            .find_map(|NodeId(bb)| self.final_name_assginments[&bb].1.get(&reg).copied())
            .unwrap()
    }

    fn make_ssa(&mut self, mir: &'a mir::Mir) {
        self.visit(mir.start);
        while let Some(action) = self.queue.pop_front() {
            match action {
                Action::MakeBlockSsa(block_id) => self.make_block_ssa(mir, block_id),
                Action::CommitBlock {
                    old_id: old_block_id,
                    block,
                    ref nr,
                    term,
                } => {
                    let term = match term {
                        mir::Terminator::Jump(next) => {
                            mir::Terminator::Jump(self.create_block_ref(next.id, nr))
                        }
                        mir::Terminator::If {
                            cond,
                            if_true,
                            if_false,
                        } => mir::Terminator::If {
                            cond: self.resolve(nr, old_block_id, *cond),
                            if_true: self.create_block_ref(if_true.id, nr),
                            if_false: self.create_block_ref(if_false.id, nr),
                        },
                        mir::Terminator::ProgramExit => mir::Terminator::ProgramExit,
                    };

                    self.builder.commit(block, term);
                }
            }
        }
    }

    fn make_block_ssa(&mut self, mir: &'a mir::Mir, block_id: mir::BasicBlockId) {
        let mut nr = HashMap::new();

        let mut block_args = Vec::new();

        if let Some(args) = self.block_args.get(&block_id) {
            block_args.reserve(args.len());
            for &(reg, new_reg) in args {
                nr.insert(reg, new_reg);
                block_args.push(new_reg);
            }
        }

        let old_block = &mir.blocks[&block_id];
        let (new_id, mut block) = self.builder.new_block();

        let mut regs = self.final_name_assginments[&block_id].0.clone();

        for &instr in &old_block.instrs {
            block.instrs.push(match instr {
                // drop elaboration should run before conversion to ssa
                mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,

                mir::Instr::ConsolePrint(val) => {
                    mir::Instr::ConsolePrint(self.resolve(&nr, block_id, val))
                }
                mir::Instr::ConsoleInput(reg) => {
                    let new_reg = regs.create();
                    nr.insert(reg, new_reg);
                    mir::Instr::ConsoleInput(new_reg)
                }
                mir::Instr::Store { dest, val } => {
                    let new_reg = regs.create();

                    let val = self.resolve(&nr, block_id, val);
                    nr.insert(dest, new_reg);
                    mir::Instr::Store { dest: new_reg, val }
                }
                mir::Instr::BinOp {
                    op,
                    dest,
                    left,
                    right,
                } => {
                    let left = self.resolve(&nr, block_id, left);
                    let right = self.resolve(&nr, block_id, right);
                    let new_reg = regs.create();
                    nr.insert(dest, new_reg);
                    mir::Instr::BinOp {
                        op,
                        dest: new_reg,
                        left,
                        right,
                    }
                }
            })
        }

        self.block_mapping.insert(block_id, new_id);

        match &old_block.term {
            mir::Terminator::Jump(next) => self.visit(next.id),
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                self.visit(if_true.id);
                self.visit(if_false.id);
            }
            mir::Terminator::ProgramExit => (),
        }

        block.args = block_args;
        self.queue.push_back(Action::CommitBlock {
            old_id: block_id,
            block,
            nr,
            term: &old_block.term,
        });
    }

    fn create_block_ref(&self, id: mir::BasicBlockId, nr: &Reg2Reg) -> mir::BasicBlockRef {
        let args = if let Some(args) = self.block_args.get(&id) {
            let mut new_args = Vec::new();
            for &(arg, _new_reg) in args {
                new_args.push(mir::Val::Reg(self.resolve_register(nr, id, arg)));
            }
            new_args
        } else {
            Vec::new()
        };
        let block_id = self.block_mapping[&id];

        mir::BasicBlockRef { id: block_id, args }
    }
}

pub fn to_ssa(mir: &mir::Mir) -> mir::Mir {
    assert!(!mir.is_ssa);

    let mut variables = HashMap::new();
    let mut regs = mir::RegAllocator::new();

    let mut predecessors = HashMap::new();
    predecessors.insert(mir.start, Vec::new());
    let mut add_pred = |child: mir::BasicBlockId, pred: mir::BasicBlockId| {
        predecessors
            .entry(child)
            .or_insert_with(Vec::new)
            .push(pred);
    };

    let mut max_block_id = mir::BasicBlockId::normal_start();

    // stablize output
    let mut blocks = Vec::from_iter(
        mir.blocks
            .iter()
            .map(|(&block_id, block)| (block_id, block)),
    );
    blocks.sort_unstable();

    for (block_id, block) in blocks {
        let block_regs = regs.clone();
        let mut block_vars = HashMap::new();
        for &instr in &block.instrs {
            match instr {
                // drop elaboration should run before conversion to ssa
                mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,
                // no write targets
                mir::Instr::ConsolePrint(_) => continue,
                //
                mir::Instr::ConsoleInput(reg)
                | mir::Instr::Store { dest: reg, val: _ }
                | mir::Instr::BinOp {
                    op: _,
                    dest: reg,
                    left: _,
                    right: _,
                } => {
                    let new_reg = regs.create();
                    block_vars.insert(reg, new_reg);
                }
            }
        }

        variables.insert(block_id, (block_regs, block_vars));

        max_block_id = max_block_id.max(block_id);
        match &block.term {
            mir::Terminator::Jump(next) => add_pred(next.id, block_id),
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                add_pred(if_true.id, block_id);
                add_pred(if_false.id, block_id);
            }
            mir::Terminator::ProgramExit => (),
        }
    }

    let predecessors = &predecessors;

    let graph = MirGraph {
        mir,
        max: max_block_id,
        predecessors,
    };

    let dominators = &petgraph::algo::dominators::simple_fast(graph, NodeId(mir.start));
    let dom_frontier = &dominator_frontier(mir, predecessors, dominators);
    let block_args = &calculate_block_args(mir, dom_frontier, &mut regs);

    for (block_id, args) in block_args {
        let (_regs, variables) = variables.get_mut(block_id).unwrap();
        for &(arg, new_reg) in args {
            variables.entry(arg).or_insert(new_reg);
        }
    }

    let mut ssa_builder = SsaBuilder {
        final_name_assginments: &variables,
        dominators,

        block_args,
        block_mapping: BlockMap::default(),

        builder: mir::MirBuilder::default(),
        queue: VecDeque::new(),
        visited: graph.visit_map(),
    };

    ssa_builder.make_ssa(mir);

    ssa_builder.builder.finish()
}

enum Action<'a> {
    MakeBlockSsa(mir::BasicBlockId),
    CommitBlock {
        old_id: mir::BasicBlockId,
        block: mir::BasicBlockBuilder,
        nr: Reg2Reg,
        term: &'a mir::Terminator,
    },
}

fn calculate_block_args(
    mir: &mir::Mir,
    dom_frontier: &BlockMap<Vec<mir::BasicBlockId>>,
    regs: &mut mir::RegAllocator,
) -> BlockMap<Vec<(mir::Reg, mir::Reg)>> {
    // algorithm adapted from
    // https://pages.cs.wisc.edu/~fischer/cs701.f14/lectures/L10.pdf
    // page 18/34 (or 191 in bottom right corner)

    let mut variables = HashSet::new();

    let mut block_args = BlockMap::default();
    let mut initial = HashMap::new();

    for (&block_id, block) in &mir.blocks {
        for &instr in block.instrs.iter().rev() {
            match instr {
                mir::Instr::EndLifetime(_) | mir::Instr::ConsolePrint(_) => (),
                mir::Instr::ConsoleInput(dest)
                | mir::Instr::Store { dest, val: _ }
                | mir::Instr::BinOp {
                    op: _,
                    dest,
                    left: _,
                    right: _,
                } => {
                    initial
                        .entry(dest)
                        .or_insert_with(HashSet::new)
                        .insert(block_id);
                }
                mir::Instr::StartLifetime(reg) => {
                    // any registers created in this node can't be lifted to phi nodes
                    // since they aren't defined before this node

                    if let Some(initial) = initial.get_mut(&reg) {
                        initial.remove(&block_id);
                    }
                    variables.insert(reg);
                }
            }
            if let mir::Instr::StartLifetime(reg) = instr {
                variables.insert(reg);
            }
        }
    }

    let mut stack = Vec::new();

    for reg in variables {
        stack.clear();

        let Some(initial) = initial.get(&reg) else {
            continue;
        };
        stack.extend(initial);

        while let Some(block_id) = stack.pop() {
            let dom_frontier = dom_frontier
                .get(&block_id)
                .map(Vec::as_slice)
                .unwrap_or_default();

            for &block_id in dom_frontier {
                let block_args = block_args.entry(block_id).or_insert_with(Vec::new);

                if block_args.iter().any(|(r, _)| *r == reg) {
                    continue;
                }

                block_args.push((reg, regs.create()));

                if initial.contains(&block_id) {
                    continue;
                }

                stack.push(block_id);
            }
        }
    }

    block_args
}

fn dominator_frontier(
    mir: &mir::Mir,
    predecessors: &HashMap<mir::BasicBlockId, Vec<mir::BasicBlockId>>,
    dominators: &petgraph::algo::dominators::Dominators<NodeId>,
) -> BlockMap<Vec<mir::BasicBlockId>> {
    // https://en.wikipedia.org/wiki/Static_single-assignment_form#Computing_minimal_SSA_using_dominance_frontiers

    let mut dom_frontier = HashMap::new();

    for &block_id in mir.blocks.keys() {
        let pred = &predecessors[&block_id][..];

        if pred.len() < 2 {
            continue;
        }

        let dom = dominators.immediate_dominator(NodeId(block_id)).unwrap().0;

        for mut runner in pred.iter().copied() {
            while runner != dom {
                let frontier = dom_frontier.entry(runner).or_insert_with(Vec::new);
                if !frontier.contains(&block_id) {
                    frontier.push(block_id);
                }

                runner = dominators.immediate_dominator(NodeId(runner)).unwrap().0;
            }
        }
    }

    dom_frontier
}
