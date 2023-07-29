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

pub fn to_ssa(mir: &mir::Mir) -> mir::Mir {
    assert!(!mir.is_ssa);

    let mut max_block_id = mir::BasicBlockId::normal_start();

    for &id in mir.blocks.keys() {
        max_block_id = max_block_id.max(id);
    }

    let mut queue = VecDeque::new();

    let graph = MirGraph {
        mir,
        max: max_block_id,
    };

    queue.push_back(0);

    let mut predessors = HashMap::new();

    predessors.insert(mir.start, Vec::new());
    for NodeId(node) in graph.node_identifiers() {
        for NodeId(child) in graph.neighbors(NodeId(node)) {
            predessors.entry(child).or_insert_with(Vec::new).push(node);
        }
    }

    let dominators = petgraph::algo::dominators::simple_fast(graph, NodeId(mir.start));

    let mut dom_frontier = HashMap::new();

    // https://en.wikipedia.org/wiki/Static_single-assignment_form#Computing_minimal_SSA_using_dominance_frontiers
    for &id in mir.blocks.keys() {
        let predessors = &predessors[&id][..];

        if predessors.len() < 2 {
            continue;
        }

        for &predecessor in predessors {
            let mut runner = predecessor;
            let NodeId(imm_dom) = dominators.immediate_dominator(NodeId(id)).unwrap();

            while runner != imm_dom {
                dom_frontier
                    .entry(runner)
                    .or_insert_with(HashSet::new)
                    .insert(id);

                runner = dominators.immediate_dominator(NodeId(runner)).unwrap().0;
            }
        }
    }

    let mut builder = mir::MirBuilder::default();
    let mut regs = &mut mir::RegAllocator::new();

    let mut queue = VecDeque::new();
    let mut visited = graph.visit_map();
    queue.push_back(Action::Process(mir.start));
    visited.insert(mir.start.to_usize());

    assert!(!petgraph::algo::is_cyclic_directed(graph)); // for now we don't handle loops

    type Reg2Reg = HashMap<mir::Reg, mir::Reg>;
    type BlockArgs = Vec<(mir::Reg, Vec<(mir::BasicBlockId, mir::Reg)>)>;
    type BlockMap<T> = HashMap<mir::BasicBlockId, T>;

    let mut names = BlockMap::<Reg2Reg>::new();
    let mut block_args = BlockMap::<BlockArgs>::new();
    let mut block_mapping = HashMap::new();

    let mut resolve_visited = HashSet::new();
    let mut resolve_stack = Vec::new();

    while let Some(action) = queue.pop_front() {
        let mut visit = |next: mir::BasicBlockId| {
            if !visited.contains(next.to_usize()) {
                visited.insert(next.to_usize());
                queue.push_back(Action::Process(next));
            }
        };

        let mut resolve = |block_id,
                           names: &BlockMap<Reg2Reg>,
                           block_args: &mut BlockArgs,
                           regs: &mut mir::RegAllocator,
                           nr: &mut _,
                           val: mir::Val|
         -> mir::Val {
            let resolve = |nr: &mut Reg2Reg, reg: mir::Reg| -> mir::Reg {
                if let Some(&name) = nr.get(&reg) {
                    // FIXME: with loops we may miss other sources
                    return name;
                }

                let NodeId(dom) = dominators.immediate_dominator(NodeId(block_id)).unwrap();
                let pred = &predessors[&block_id][..];

                let mut of_dominator = None;
                let mut other_branches = Vec::new();
                resolve_visited.clear();
                resolve_visited.extend(pred);
                resolve_stack.extend(pred.iter().copied().zip(core::iter::repeat(false)));

                // use a fix-point search up to the dominator to capture all
                // possible sources for reg.
                while let Some((bb, is_dom)) = resolve_stack.pop() {
                    let x = names[&bb].get(&reg).copied();
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
                        // but if the register isn't found in the current dominator
                        // we can skip to the previous dominator
                        // we can guaranteed that the reg won't be set in any non-dominating
                        // block of the dominator since that would imply that we placed a block arg
                        // incorrectly, which can't happen by induction (this is the inductive case)
                        let dom = dominators.immediate_dominator(NodeId(block_id)).unwrap().0;
                        resolve_stack.push((dom, true));
                    } else {
                        // if the node isn't a dominator, then walk the predecessors to collect any
                        // assignments in there too
                        // this is one base case

                        for &bb in &predessors[&block_id][..] {
                            if resolve_visited.insert(bb) {
                                resolve_stack.push((bb, false));
                            }
                        }
                    }
                }

                let (dom, dom_reg) = of_dominator.unwrap();

                if other_branches.is_empty() {
                    dom_reg
                } else {
                    let reg = regs.create();

                    other_branches.push((dom, dom_reg));
                    block_args.push((reg, other_branches));

                    reg
                }
            };

            match val {
                mir::Val::ConstI32(_) | mir::Val::ConstBool(_) => val,
                mir::Val::Reg(reg) => mir::Val::Reg({ resolve }(nr, reg)),
            }
        };

        match action {
            Action::Process(block_id) => {
                let mut current_block_args = BlockArgs::default();
                let mut resolve = |nr: &mut _, regs: &mut mir::RegAllocator, val| {
                    resolve(block_id, &names, &mut current_block_args, regs, nr, val)
                };

                let mut nr = HashMap::new();

                let old_block = &mir.blocks[&block_id];
                let (new_id, mut block) = builder.new_block();

                for &instr in &old_block.instrs {
                    block.instrs.push(match instr {
                        // drop elaboration should run before conversion to ssa
                        mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,

                        mir::Instr::ConsolePrint(val) => {
                            mir::Instr::ConsolePrint(resolve(&mut nr, regs, val))
                        }
                        mir::Instr::ConsoleInput(reg) => {
                            let new_reg = regs.create();
                            nr.insert(reg, new_reg);
                            mir::Instr::ConsoleInput(new_reg)
                        }
                        mir::Instr::Store { dest, val } => {
                            let new_reg = regs.create();

                            let val = resolve(&mut nr, regs, val);
                            nr.insert(dest, new_reg);
                            mir::Instr::Store { dest: new_reg, val }
                        }
                        mir::Instr::BinOp {
                            op,
                            dest,
                            left,
                            right,
                        } => {
                            let left = resolve(&mut nr, regs, left);
                            let right = resolve(&mut nr, regs, right);
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

                block_mapping.insert(block_id, new_id);

                match &old_block.term {
                    mir::Terminator::Jump(next) => visit(next.id),
                    mir::Terminator::If {
                        cond: _,
                        if_true,
                        if_false,
                    } => {
                        visit(if_true.id);
                        visit(if_false.id);
                    }
                    mir::Terminator::ProgramExit => (),
                }

                names.insert(block_id, nr);
                block_args.insert(block_id, current_block_args);
                queue.push_back(Action::Commit(new_id, block, &old_block.term));
            }
            Action::Commit(block_id, block, term) => {
                let mut nr = names.remove(&block_id).unwrap();
                let mut current_block_args = block_args.remove(&block_id).unwrap();
                let term = match term {
                    mir::Terminator::Jump(next) => {
                        mir::Terminator::Jump(mir::BasicBlockRef::new(block_mapping[&next.id]))
                    }
                    mir::Terminator::If {
                        cond,
                        if_true,
                        if_false,
                    } => {
                        let cond = resolve(
                            block_id,
                            &names,
                            &mut current_block_args,
                            regs,
                            &mut nr,
                            *cond,
                        );
                        mir::Terminator::If {
                            cond,
                            if_true: mir::BasicBlockRef::new(block_mapping[&if_true.id]),
                            if_false: mir::BasicBlockRef::new(block_mapping[&if_false.id]),
                        }
                    }
                    mir::Terminator::ProgramExit => mir::Terminator::ProgramExit,
                };

                names.insert(block_id, nr);
                if !current_block_args.is_empty() {
                    block_args.insert(block_id, current_block_args);
                }
                builder.commit(block, term);
            }
        }
    }

    for (block_id, block_args) in block_args {
        let block_id = block_mapping[&block_id];
        let bb = builder.blocks.get_mut(&block_id).unwrap();

        for &(reg, _) in &block_args {
            bb.args.push(Some(reg));
        }

        for (_, sources) in block_args {
            for (other_block_id, source_reg) in sources {
                let bb = builder
                    .blocks
                    .get_mut(&block_mapping[&other_block_id])
                    .unwrap();

                match &mut bb.term {
                    mir::Terminator::Jump(next) => {
                        assert_eq!(next.id, block_id);
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

    builder.finish()
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
