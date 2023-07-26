use std::{mem::MaybeUninit, ops::Deref, sync::Once};

use istr::IStr;

static ONCE: Once = Once::new();
static mut KEYWORDS: MaybeUninit<KeywordInfo> = MaybeUninit::uninit();

pub struct KeywordInfo {
    kws: istr::IStrMap<Keyword>,
    pub block_kw: IStr,
    pub named_block_kw: IStr,
}

#[derive(Debug, Clone, Copy)]
pub enum Keyword {
    Block,
    NamedBlock,
    Let,
    Set,
    Get,
    If,
    Loop,
    Break,
    Continue,
    Input,
    Print,
    Add,
    Sub,
    Eq,
    DebugBlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Keywords(());

impl Keywords {
    pub fn init() -> Self {
        ONCE.call_once(|| unsafe {
            KEYWORDS = MaybeUninit::new(KeywordInfo {
                block_kw: IStr::new("block"),
                named_block_kw: IStr::new("named-block"),
                kws: std::iter::FromIterator::from_iter([
                    (IStr::new("block"), Keyword::Block),
                    (IStr::new("debug-block"), Keyword::DebugBlock),
                    (IStr::new("named-block"), Keyword::NamedBlock),
                    (IStr::new("let"), Keyword::Let),
                    (IStr::new("set"), Keyword::Set),
                    (IStr::new("get"), Keyword::Get),
                    (IStr::new("if"), Keyword::If),
                    (IStr::new("loop"), Keyword::Loop),
                    (IStr::new("break"), Keyword::Break),
                    (IStr::new("continue"), Keyword::Continue),
                    (IStr::new("input"), Keyword::Input),
                    (IStr::new("print"), Keyword::Print),
                    (IStr::new("+"), Keyword::Add),
                    (IStr::new("-"), Keyword::Sub),
                    (IStr::new("="), Keyword::Eq),
                ]),
            })
        });

        Self(())
    }

    pub fn get(self, kw: IStr) -> Option<Keyword> {
        self.kws.get(&kw).copied()
    }
}

impl Deref for Keywords {
    type Target = KeywordInfo;

    fn deref(&self) -> &Self::Target {
        unsafe { &*KEYWORDS.as_ptr() }
    }
}
