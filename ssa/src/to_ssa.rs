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
    let mut regs = mir::RegAllocator::new();

    let mut queue = VecDeque::new();
    let mut visited = graph.visit_map();
    queue.push_back(Action::Process(mir.start));
    visited.insert(mir.start.to_usize());

    assert!(!petgraph::algo::is_cyclic_directed(graph)); // for now we don't handle loops

    let mut names = HashMap::<mir::BasicBlockId, HashMap<mir::Reg, mir::Reg>>::new();

    while let Some(action) = queue.pop_front() {
        dbg!(&action);

        let mut visit = |next: mir::BasicBlockId| {
            if !visited.contains(next.to_usize()) {
                visited.insert(next.to_usize());
                queue.push_back(Action::Process(next));
            }
        };

        match action {
            Action::Process(block_id) => {
                let resolve = |nr: &mut HashMap<mir::Reg, mir::Reg>, reg: mir::Reg| -> mir::Reg {
                    if let Some(&name) = nr.get(&reg) {
                        return name;
                    }

                    let NodeId(dom) = dominators.immediate_dominator(NodeId(block_id)).unwrap();
                    let pred = &predessors[&block_id][..];

                    dbg!(dom);
                    dbg!(&names);
                    dbg!(pred);

                    let mut of_dominator = None;

                    for &bb in pred {
                        dbg!(bb);
                        let x = names[&bb].get(&reg).copied();

                        if bb == dom {
                            of_dominator = Some(x.unwrap());
                        } else {
                            //
                        }

                        dbg!((bb, x, bb == dom));
                    }

                    todo!()
                };

                let resolve = |nr: &mut _, val: mir::Val| -> mir::Val {
                    match val {
                        mir::Val::ConstI32(_) | mir::Val::ConstBool(_) => val,
                        mir::Val::Reg(reg) => mir::Val::Reg(resolve(nr, reg)),
                    }
                };

                let mut nr = HashMap::new();
                let old_block = &mir.blocks[&block_id];
                let (new_id, mut block) = builder.new_block();

                for &instr in &old_block.instrs {
                    block.instrs.push(match instr {
                        // drop elaboration should run before conversion to ssa
                        mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,

                        mir::Instr::ConsolePrint(val) => {
                            mir::Instr::ConsolePrint(resolve(&mut nr, val))
                        }
                        mir::Instr::ConsoleInput(reg) => {
                            let new_reg = regs.create();
                            nr.insert(reg, new_reg);
                            mir::Instr::ConsoleInput(new_reg)
                        }
                        mir::Instr::Store { dest, val } => {
                            let new_reg = regs.create();

                            let val = match val {
                                mir::Val::ConstI32(_) | mir::Val::ConstBool(_) => val,
                                mir::Val::Reg(_) => todo!(),
                            };
                            nr.insert(dest, new_reg);
                            mir::Instr::Store { dest: new_reg, val }
                        }
                        mir::Instr::BinOp {
                            op,
                            dest,
                            left,
                            right,
                        } => {
                            let new_reg = regs.create();
                            nr.insert(dest, new_reg);
                            mir::Instr::BinOp {
                                op,
                                dest: new_reg,
                                left: resolve(&mut nr, left),
                                right: resolve(&mut nr, right),
                            }
                        }
                    })
                }

                dbg!(&nr);

                names.insert(block_id, nr);

                match &old_block.term {
                    mir::Terminator::Jump(next) => visit(next.id),
                    mir::Terminator::If {
                        cond: _,
                        if_true,
                        if_false,
                    } => {
                        visit(if_true.id);
                        visit(if_false.id);
                        queue.push_back(Action::Commit(block_id, block));
                    }
                    mir::Terminator::ProgramExit => todo!(),
                }
            }
            Action::Commit(_, _) => todo!(),
        }
    }

    dbg!(dom_frontier);

    todo!()
}

enum Action {
    Process(mir::BasicBlockId),
    Commit(mir::BasicBlockId, mir::BasicBlockBuilder),
}

impl Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process(arg0) => f.debug_tuple("Process").field(arg0).finish(),
            Self::Commit(arg0, _arg1) => f.debug_tuple("Commit").field(arg0).finish(),
        }
    }
}
