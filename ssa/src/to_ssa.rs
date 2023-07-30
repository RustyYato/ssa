use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;

use crate::{
    aliases::*,
    graph::{MirGraph, NodeId},
    mir,
};

use petgraph::visit::Visitable;

struct SsaBuilder<'a> {
    /// the final name assignments tell us where each register in the original
    /// MIR ended up in the SSA-MIR at the end of each block.
    /// and since each block doesn't have any control flow until the terminator
    /// when resolving names, we only need to look at the final assignments to know
    /// if a block contains the unresolved name
    block_reg_info: &'a BlockMap<BlockNames>,

    /// When resolving nodes, we only need to look at the current block's dominators.
    dominators: &'a petgraph::algo::dominators::Dominators<NodeId>,

    /// The block arguments for each block that needs them, used to reconcile
    /// conditional mutation.
    block_args: &'a BlockMap<Vec<(mir::Reg, mir::Reg)>>,

    /// The new SSA-MIR
    builder: mir::MirBuilder,

    /// A mapping of old block ids to the corrosponding new block id
    block_mapping: BlockMap<mir::BasicBlockId>,

    /// A queue of actions, use to avoid recursion
    /// An action either converts a block to SSA or commits a block after all
    /// of it's children have been converted to SSA
    queue: VecDeque<Action<'a>>,

    // The visited blocks while converting them to ssa
    visited: fixedbitset::FixedBitSet,
}

struct BlockNames {
    /// The register allocator at the start of the block
    ///
    /// This allows reconstructing exactly which registers are
    /// allocated to each node, as long as they are done in the
    /// same order in `Extractor::extract` and in `SsaBuilder::make_block_ssa`
    regs: mir::RegAllocator,

    /// The ssa-register assigned to each register written to in the block
    /// If a register is written to multiple times, the last ssa-register
    /// will be kept
    vars: Reg2Reg,
}

struct Extractor {
    regs: mir::RegAllocator,
    // see `SsaBuilder.block_reg_info`
    block_reg_info: BlockMap<BlockNames>,
    // used to initialize `SsaBuilder.visited` to a big enough size
    // to hold all blocks in the original MIR
    max_block_id: mir::BasicBlockId,
    // the predecessor of each block
    predecessors: BlockMap<Vec<mir::BasicBlockId>>,
}

impl Extractor {
    /// Extract out...
    /// * the final ssa-registers that each register maps to
    /// * the maximum block id
    /// * the predecessor of this block
    ///
    /// This is done in one function, so we can extract all of this information in
    /// one pass of the MIR.
    fn extract(&mut self, block_id: mir::BasicBlockId, block: &mir::BasicBlock) {
        let block_regs = self.regs.clone();
        let mut block_vars = HashMap::new();
        for &instr in &block.instrs {
            match instr {
                // drop elaboration should run before conversion to ssa
                mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,
                // no write targets
                mir::Instr::ConsolePrint(_) => continue,
                // one write target, create a new register
                // to ensure that the each register is written to at most once
                mir::Instr::ConsoleInput(reg)
                | mir::Instr::WriteUninit(reg)
                | mir::Instr::Store { dest: reg, val: _ }
                | mir::Instr::BinOp {
                    op: _,
                    dest: reg,
                    left: _,
                    right: _,
                } => {
                    let new_reg = self.regs.create();
                    block_vars.insert(reg, new_reg);
                }
            }
        }

        // see SsaBuilder.final_name_assignments for details
        self.block_reg_info.insert(
            block_id,
            BlockNames {
                regs: block_regs,
                vars: block_vars,
            },
        );

        // track the max_block_id so that the fixedbitset can be initialized to the correct size
        self.max_block_id = self.max_block_id.max(block_id);

        // track predecessor relationships, to build the dominator frontier
        match &block.term {
            mir::Terminator::Jump(next) => self.track_predecessor(next.id, block_id),
            mir::Terminator::If {
                cond: _,
                if_true,
                if_false,
            } => {
                self.track_predecessor(if_true.id, block_id);
                self.track_predecessor(if_false.id, block_id);
            }
            mir::Terminator::ProgramExit => (),
        }
    }

    fn track_predecessor(&mut self, child: mir::BasicBlockId, pred: mir::BasicBlockId) {
        self.predecessors
            .entry(child)
            .or_insert_with(Vec::new)
            .push(pred);
    }
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
        // if the node is written to in the current block, then just reuse that ssa-register
        if let Some(&reg) = nr.get(&reg) {
            return reg;
        }

        // We only need to look at the dominators instead of all parents because
        // any parent which doesn't strictly dominate this node has at most a conditional mutation
        // if the parent is after immediate dominator, then that conditioanl mutation is reconciled
        // as a block arg on the current node, which is inserted into `nr`
        // if the parent is before the immediate dominator, then the dominator contains a block arg
        // which will be inserted into `final_name_assginments` for that block
        // if there is no conditional mutation, then by definition the variable must be written to
        // in a dominator.
        // Finally, it is illegal to pass a MIR where any register would be unresolved (i.e. was used before it was defined)

        self.dominators
            .strict_dominators(NodeId(block_id))
            .unwrap()
            .find_map(|NodeId(bb)| self.block_reg_info[&bb].vars.get(&reg).copied())
            .unwrap()
    }

    fn make_ssa(&mut self, mir: &'a mir::Mir) {
        // recurse through all the blocks starting with the start block
        // visiting all blocks in pre-order traversal, and committing
        // the blocks in a post-order traversal in the same loop
        self.visit(mir.start);
        while let Some(action) = self.queue.pop_front() {
            match action {
                // this will append an action to make each child ssa and an action to commit the `block_id`
                Action::MakeBlockSsa(block_id) => self.make_block_ssa(mir, block_id),

                // Commit the block, and create reconcile block args
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

        let mut regs = self.block_reg_info[&block_id].regs.clone();

        for &instr in &old_block.instrs {
            block.instrs.push(match instr {
                // drop elaboration should run before conversion to ssa but should keep these lifetime annotations in place
                mir::Instr::StartLifetime(_) | mir::Instr::EndLifetime(_) => continue,

                mir::Instr::ConsolePrint(val) => {
                    mir::Instr::ConsolePrint(self.resolve(&nr, block_id, val))
                }

                // NOTE: the new registers should be created and inserted *after*
                // we resolve the values, otherwise the values could in theory refer to
                // their own definition which wouldn't work out
                mir::Instr::ConsoleInput(reg) => {
                    let new_reg = regs.create();
                    nr.insert(reg, new_reg);
                    mir::Instr::ConsoleInput(new_reg)
                }
                mir::Instr::WriteUninit(reg) => {
                    let new_reg = regs.create();
                    nr.insert(reg, new_reg);
                    mir::Instr::WriteUninit(new_reg)
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

    fn create_block_ref(&self, id: mir::BasicBlockId, nr: &Reg2Reg) -> mir::JumpTarget {
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

        mir::JumpTarget { id: block_id, args }
    }
}

pub fn to_ssa(mir: &mir::Mir) -> mir::Mir {
    to_ssa_::<false>(mir)
}

pub fn to_ssa_stable(mir: &mir::Mir) -> mir::Mir {
    to_ssa_::<true>(mir)
}

fn to_ssa_<const STABLE_OUTPUT: bool>(mir: &mir::Mir) -> mir::Mir {
    assert!(!mir.is_ssa);

    // stablize output
    let blocks = mir
        .blocks
        .iter()
        .map(|(&block_id, block)| (block_id, block));

    let mut extractor = Extractor {
        regs: mir::RegAllocator::new(),
        block_reg_info: BlockMap::default(),
        max_block_id: mir.start,
        predecessors: BlockMap::default(),
    };

    let blocks = if STABLE_OUTPUT {
        let mut blocks = Vec::from_iter(blocks);
        blocks.sort_unstable();
        either::Left(blocks.into_iter())
    } else {
        either::Right(blocks)
    };

    extractor.predecessors.insert(mir.start, Vec::new());
    blocks.for_each(|(block_id, block)| extractor.extract(block_id, block));

    // Construct a graph represetnting the mir, which it compatible with petgraph's Graph API
    let graph = MirGraph {
        mir,
        max: extractor.max_block_id,
        predecessors: &extractor.predecessors,
    };

    let dominators = &petgraph::algo::dominators::simple_fast(graph, NodeId(mir.start));
    // the dominator frontier contains all sucessors which are not dominated the keyed block
    // this tells us where to put block args, since these are exactly the locations where different
    // conditional mutations must be reconciled
    let dom_frontier = &dominator_frontier(mir, &extractor.predecessors, dominators);
    // calculate the block arguments for all blocks, if needed
    // this reconciled conditional mutations to the same register in an SSA friendly way
    let block_args = &calculate_block_args::<STABLE_OUTPUT>(mir, dom_frontier, &mut extractor.regs);

    // Add block args to the final name assignments so that we can resolve to them
    // This is critical to correctly reconcile loops
    for (block_id, args) in block_args {
        let block = extractor.block_reg_info.get_mut(block_id).unwrap();
        for &(arg, new_reg) in args {
            block.vars.entry(arg).or_insert(new_reg);
        }
    }

    let mut ssa_builder = SsaBuilder {
        block_reg_info: &extractor.block_reg_info,
        dominators,
        block_args,

        block_mapping: BlockMap::default(),
        builder: mir::MirBuilder::default(),

        queue: VecDeque::new(),
        visited: graph.visit_map(),
    };

    ssa_builder.make_ssa(mir);

    let mut mir = ssa_builder.builder.finish();
    mir.is_ssa = true;
    mir
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

fn calculate_block_args<const STABLE_OUTPUT: bool>(
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
                | mir::Instr::WriteUninit(dest)
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
                    // any registers created in this node can't be lifted to block args
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
        if STABLE_OUTPUT {
            stack.sort_unstable();
        }

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
