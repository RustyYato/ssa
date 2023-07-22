use std::str::FromStr;

pub use crate::parser::ParseError;

pub struct Syntax {
    pub name: istr::IBytes,
    pub args: Vec<Syntax>,
}

impl core::fmt::Debug for Syntax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Syntax")
            .field("name", &bstr::BStr::new(self.name.to_bytes()))
            .field("args", &self.args)
            .finish()
    }
}

impl FromStr for Syntax {
    type Err = ParseError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_bytes(s.as_bytes())
    }
}

impl Syntax {
    #[inline]
    pub fn from_bytes(text: &[u8]) -> Result<Syntax, ParseError> {
        crate::parser::parse(text)
    }
}
