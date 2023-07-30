use crate::{aliases::*, mir};

use petgraph::{adj, data, visit};

#[derive(Clone, Copy)]
pub(crate) struct MirGraph<'a> {
    pub(crate) mir: &'a mir::Mir,
    pub(crate) max: mir::BasicBlockId,
    pub(crate) predecessors: &'a BlockMap<Vec<mir::BasicBlockId>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(pub(crate) mir::BasicBlockId);

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

pub(crate) struct Nodes<'a> {
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
