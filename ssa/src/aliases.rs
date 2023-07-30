use hashbrown::{HashMap, HashSet};

use crate::mir;

pub type RegMap<T> = HashMap<mir::Reg, T>;
pub type BlockMap<T> = HashMap<mir::BasicBlockId, T>;
pub type RegSet = HashSet<mir::Reg>;

pub type Reg2Reg = RegMap<mir::Reg>;
