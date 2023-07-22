use crate::syntax::Syntax;

pub struct Parser<'text> {
    text: &'text [u8],
}

#[derive(Debug)]
pub enum ParseError {
    ExpectedWord(u8),
    ExpectedWordEof,
    ExpectedEof(u8),
}

impl std::error::Error for ParseError {}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ParseError::ExpectedWord(c) => {
                if c.is_ascii() {
                    write!(
                        f,
                        "expected a word, but found a non-graphic or reserverd character: {}",
                        c as char
                    )
                } else {
                    write!(
                        f,
                        "expected a word, but found a non-graphic or reserverd character: 0x{c:x}"
                    )
                }
            }
            ParseError::ExpectedEof(c) => {
                if c.is_ascii() {
                    write!(
                        f,
                        "expected EOF, but found a non-graphic or reserverd character: {}",
                        c as char
                    )
                } else {
                    write!(
                        f,
                        "expected EOF, but found a non-graphic or reserverd character: 0x{c:x}"
                    )
                }
            }
            ParseError::ExpectedWordEof => write!(f, "expected a word, but found EOF"),
        }
    }
}

type Result<T, E = ParseError> = std::result::Result<T, E>;

pub fn parse(text: &[u8]) -> Result<Syntax> {
    let mut parser = Parser::new(text);

    let syn = parser.parse()?;

    if let [b, ..] = parser.text {
        return Err(ParseError::ExpectedEof(*b));
    }

    Ok(syn)
}

impl<'text> Parser<'text> {
    pub fn new(text: &'text [u8]) -> Self {
        Self { text }
    }

    pub fn parse_ws(&mut self) {
        let mut bytes = self.text;

        while let [first, rest @ ..] = bytes {
            if first.is_ascii_whitespace() {
                bytes = rest;
            } else {
                break;
            }
        }

        self.text = bytes
    }

    pub fn parse_word(&mut self) -> Result<istr::IStr> {
        let mut bytes = self.text;

        if self.text.is_empty() {
            return Err(ParseError::ExpectedWordEof);
        }

        loop {
            match bytes {
                [b @ (b'(' | b')' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' '), ..] => {
                    if self.text.len() == bytes.len() {
                        return Err(ParseError::ExpectedWord(*b));
                    } else {
                        break;
                    }
                }
                [b, rest @ ..] if b.is_ascii_graphic() => bytes = rest,
                [b, ..] => {
                    if self.text.len() == bytes.len() {
                        return Err(ParseError::ExpectedWord(*b));
                    } else {
                        break;
                    }
                }
                [] => break,
            }
        }

        let index = self.text.len() - bytes.len();

        let (word, rest) = self.text.split_at(index);
        self.text = rest;

        Ok(istr::IStr::new(unsafe {
            core::str::from_utf8_unchecked(word)
        }))
    }

    fn parse(&mut self) -> Result<Syntax> {
        self.parse_ws();
        match self.text {
            [b'(', rest @ ..] => {
                self.text = rest;

                let mut syn = Syntax {
                    name: self.parse_word()?,
                    args: Vec::new(),
                };

                loop {
                    self.parse_ws();

                    if let [b')', rest @ ..] = self.text {
                        self.text = rest;
                        break Ok(syn);
                    }

                    syn.args.push(self.parse()?)
                }
            }
            _ => Ok(Syntax {
                name: self.parse_word()?,
                args: Vec::new(),
            }),
        }
    }
}
