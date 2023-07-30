use hashbrown::HashMap;

use crate::mir;

pub type Reg2Reg = HashMap<mir::Reg, mir::Reg>;
pub type BlockMap<T> = HashMap<mir::BasicBlockId, T>;
