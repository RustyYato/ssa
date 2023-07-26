use std::str::FromStr;

pub use crate::parser::ParseError;

pub struct Syntax {
    pub name: istr::IStr,
    pub args: Vec<Syntax>,
}

impl core::fmt::Debug for Syntax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Syntax")
            .field("name", &self.name)
            .field("args", &self.args)
            .finish()
    }
}

impl core::fmt::Display for Syntax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_display(0, f)
    }
}

impl Syntax {
    fn fmt_display(&self, depth: u32, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..depth {
            f.write_str("  ")?
        }
        match self.args.len() {
            0 => write!(f, "{}", self.name),
            1 if self.args[0].args.is_empty() => {
                write!(f, "({} {})", self.name, self.args[0].name)
            }
            _ => {
                writeln!(f, "({}", self.name)?;
                let mut first = true;
                for arg in &self.args {
                    if first {
                        first = false;
                    } else {
                        f.write_str("\n")?
                    }
                    arg.fmt_display(depth + 1, f)?;
                }
                write!(f, ")")
            }
        }
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
