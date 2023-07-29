use std::{collections::VecDeque, fmt::Debug};

use hashbrown::{HashMap, HashSet};

use crate::mir;

use petgraph::{
    adj, data,
    visit::{self, IntoNeighbors, IntoNodeIdentifiers, Visitable},
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
type BlockArgs = Vec<(mir::Reg, Vec<(mir::BasicBlockId, mir::Reg)>)>;

struct SsaBuilder<'a> {
    final_name_assginments: &'a BlockMap<(mir::RegAllocator, Reg2Reg)>,
    predecessors: &'a BlockMap<Vec<mir::BasicBlockId>>,
    dominators: &'a petgraph::algo::dominators::Dominators<NodeId>,
    is_part_of_cycle: &'a fixedbitset::FixedBitSet,

    builder: mir::MirBuilder,
    regs: mir::RegAllocator,

    block_mapping: BlockMap<mir::BasicBlockId>,

    queue: VecDeque<Action<'a>>,
    visited: fixedbitset::FixedBitSet,

    resolve_visited: HashSet<mir::BasicBlockId>,
    resolve_stack: Vec<(mir::BasicBlockId, bool)>,
}

impl<'a> SsaBuilder<'a> {
    fn visit(&mut self, block_id: mir::BasicBlockId) {
        if !self.visited.contains(block_id.to_usize()) {
            self.visited.insert(block_id.to_usize());
            self.queue.push_back(Action::Process(block_id));
        }
    }

    fn resolve(
        &mut self,
        nr: &mut Reg2Reg,
        block_args: &mut BlockArgs,
        block_id: mir::BasicBlockId,
        val: mir::Val,
    ) -> mir::Val {
        let reg = match val {
            mir::Val::ConstI32(_) | mir::Val::ConstBool(_) => return val,
            mir::Val::Reg(reg) => reg,
        };

        let is_part_of_cycle = self.is_part_of_cycle.contains(block_id.to_usize());

        if let Some(&name) = nr.get(&reg) {
            return mir::Val::Reg(name);
        }

        let NodeId(dom) = self
            .dominators
            .immediate_dominator(NodeId(block_id))
            .unwrap();

        let mut of_dominator = None;
        let mut other_branches = Vec::new();
        self.resolve_visited.clear();
        self.resolve_stack.clear();

        if is_part_of_cycle {
            self.resolve_visited.insert(block_id);
            self.resolve_stack.push((block_id, false));
        }

        let pred = &self.predecessors[&block_id][..];
        self.resolve_visited.extend(pred);
        self.resolve_stack
            .extend(pred.iter().copied().zip(core::iter::repeat(false)));

        // use a fix-point search up to the dominator to capture all
        // possible sources for reg.
        while let Some((bb, is_dom)) = self.resolve_stack.pop() {
            let x = self.final_name_assginments[&bb].1.get(&reg).copied();
            let is_dom = is_dom || bb == dom;

            if let Some(x) = x {
                // if the current node contains an assignment to the regsiter
                // collect that assignment
                // this is one base case
                if is_dom {
                    of_dominator = Some((bb, x));
                } else {
                    other_branches.push((bb, x));
                }
            } else if is_dom {
                //FIXME:
                // split this out into it's own function, and call it
                //

                // but if the register isn't found in the current dominator
                // we can skip to the previous dominator
                // we can guaranteed that the reg won't be set in any non-dominating
                // block of the dominator since that would imply that we placed a block arg
                // incorrectly, which can't happen by induction (this is the inductive case)
                let dom = self.dominators.immediate_dominator(NodeId(bb)).unwrap().0;
                self.resolve_stack.push((dom, true));
            } else {
                // if the node isn't a dominator, then walk the predecessors to collect any
                // assignments in there too
                // this is one base case

                for &bb in &self.predecessors[&bb][..] {
                    if self.resolve_visited.insert(bb) {
                        self.resolve_stack.push((bb, false));
                    }
                }
            }
        }

        let (dom, dom_reg) = of_dominator.unwrap();

        mir::Val::Reg(if other_branches.is_empty() {
            dom_reg
        } else {
            let new_reg = self.regs.create();

            other_branches.push((dom, dom_reg));
            nr.insert(reg, new_reg);
            block_args.push((new_reg, other_branches));

            new_reg
        })
    }

    fn make_ssa(&mut self, mir: &'a mir::Mir) -> BlockMap<BlockArgs> {
        let mut names = HashMap::new();
        let mut block_args = HashMap::new();

        while let Some(action) = self.queue.pop_front() {
            match action {
                Action::Process(block_id) => {
                    let mut current_block_args = BlockArgs::default();

                    let mut nr = HashMap::new();

                    let old_block = &mir.blocks[&block_id];
                    let (new_id, mut block) = self.builder.new_block();

                    let mut regs = self.final_name_assginments[&block_id].0.clone();

                    for &instr in &old_block.instrs {
                        block.instrs.push(match instr {
                            // drop elaboration should run before conversion to ssa
                            mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,

                            mir::Instr::ConsolePrint(val) => mir::Instr::ConsolePrint(
                                self.resolve(&mut nr, &mut current_block_args, block_id, val),
                            ),
                            mir::Instr::ConsoleInput(reg) => {
                                let new_reg = regs.create();
                                nr.insert(reg, new_reg);
                                mir::Instr::ConsoleInput(new_reg)
                            }
                            mir::Instr::Store { dest, val } => {
                                let new_reg = regs.create();

                                let val =
                                    self.resolve(&mut nr, &mut current_block_args, block_id, val);
                                nr.insert(dest, new_reg);
                                mir::Instr::Store { dest: new_reg, val }
                            }
                            mir::Instr::BinOp {
                                op,
                                dest,
                                left,
                                right,
                            } => {
                                let left =
                                    self.resolve(&mut nr, &mut current_block_args, block_id, left);
                                let right =
                                    self.resolve(&mut nr, &mut current_block_args, block_id, right);
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

                    names.insert(new_id, nr);
                    block_args.insert(new_id, current_block_args);
                    self.queue
                        .push_back(Action::Commit(new_id, block, &old_block.term));
                }
                Action::Commit(block_id, block, term) => {
                    let mut nr = names.remove(&block_id).unwrap();
                    let mut current_block_args = block_args.remove(&block_id).unwrap();
                    let term = match term {
                        mir::Terminator::Jump(next) => mir::Terminator::Jump(
                            mir::BasicBlockRef::new(self.block_mapping[&next.id]),
                        ),
                        mir::Terminator::If {
                            cond,
                            if_true,
                            if_false,
                        } => {
                            let cond =
                                self.resolve(&mut nr, &mut current_block_args, block_id, *cond);
                            mir::Terminator::If {
                                cond,
                                if_true: mir::BasicBlockRef::new(self.block_mapping[&if_true.id]),
                                if_false: mir::BasicBlockRef::new(self.block_mapping[&if_false.id]),
                            }
                        }
                        mir::Terminator::ProgramExit => mir::Terminator::ProgramExit,
                    };

                    names.insert(block_id, nr);
                    if !current_block_args.is_empty() {
                        block_args.insert(block_id, current_block_args);
                    }
                    self.builder.commit(block, term);
                }
            }
        }

        block_args
    }

    fn insert_block_args(&mut self, block_args: BlockMap<BlockArgs>) {
        dbg!(&block_args);
        dbg!(&self.block_mapping);
        for (block_id, block_args) in block_args {
            // let block_id = self.block_mapping[&block_id];
            let bb = self.builder.blocks.get_mut(&block_id).unwrap();

            dbg!(block_id);

            for &(reg, _) in &block_args {
                bb.args.push(Some(reg));
            }

            for (_, sources) in block_args {
                for (other_block_id, source_reg) in sources {
                    let bb: &mut mir::BasicBlock = self
                        .builder
                        .blocks
                        .get_mut(&self.block_mapping[&other_block_id])
                        .unwrap();

                    match &mut bb.term {
                        mir::Terminator::Jump(next) => {
                            next.args.push(mir::Val::Reg(source_reg));
                        }
                        mir::Terminator::If {
                            cond: _,
                            if_true,
                            if_false,
                        } => {
                            if if_true.id == block_id {
                                if_true.args.push(mir::Val::Reg(source_reg));
                            } else {
                                assert_eq!(if_false.id, block_id);
                                if_false.args.push(mir::Val::Reg(source_reg));
                            }
                        }
                        mir::Terminator::ProgramExit => unreachable!(),
                    }
                }
            }
        }
    }
}

pub fn to_ssa(mir: &mir::Mir) -> mir::Mir {
    assert!(!mir.is_ssa);

    // FIXME: build SSA representation in two passes
    // First pass goes through each block and assigns registers to each of
    // the written to. These will be tracked by using the mapping (block id, instr id) -> new reg
    // to key each assignment, or we could just store (block id, Vec<new reg>)
    // where each instruction is visited in the same order
    //
    // in the second pass, use this generated list to resolve ids, potentially recursively
    // to the same block to ensure that cycles are taken care of

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
    for (&block_id, block) in mir.blocks.iter() {
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

    let variables = &variables;
    let predecessors = &predecessors;

    let graph = MirGraph {
        mir,
        max: max_block_id,
        predecessors,
    };

    let dominators = &petgraph::algo::dominators::simple_fast(graph, NodeId(mir.start));
    let scc = &petgraph::algo::kosaraju_scc(graph);

    let mut is_part_of_cycle = graph.visit_map();

    for component in scc {
        if component.len() <= 1 {
            continue;
        }

        for &NodeId(bb) in component {
            is_part_of_cycle.insert(bb.to_usize());
        }
    }

    let is_part_of_cycle = &is_part_of_cycle;

    let mut resolver = SsaBuilder {
        final_name_assginments: variables,
        predecessors,
        dominators,
        is_part_of_cycle,

        block_mapping: BlockMap::default(),

        builder: mir::MirBuilder::default(),
        regs,
        queue: VecDeque::new(),
        visited: graph.visit_map(),

        resolve_visited: HashSet::default(),
        resolve_stack: Vec::new(),
    };

    dbg!(predecessors);
    dbg!(scc);
    dbg!(dominators);
    resolver.visit(mir.start);
    let block_args = resolver.make_ssa(mir);
    resolver.insert_block_args(block_args);
    println!("{}", mir::StableDisplayMir::from(resolver.builder.finish()));

    todo!()
}

enum Action<'a> {
    Process(mir::BasicBlockId),
    Commit(
        mir::BasicBlockId,
        mir::BasicBlockBuilder,
        &'a mir::Terminator,
    ),
}

impl Debug for Action<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process(arg0) => f.debug_tuple("Process").field(arg0).finish(),
            Self::Commit(arg0, ..) => f.debug_tuple("Commit").field(arg0).finish(),
        }
    }
}
