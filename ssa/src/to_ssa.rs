use crate::{
    mir::{self, BasicBlockBuilder, MirBuilder},
    syntax::Syntax,
};

mod keywords;
mod name_resolver;

pub struct Encoder {
    keywords: keywords::Keywords,
    mir: MirBuilder,
    nr: name_resolver::NameResolver,
    regs: mir::RegAllocator,
}

#[derive(Debug)]
pub enum EncodingError {
    MissingArgsForPrint,
    TooManyArgsForPrint,
    MissingArgsForInput,
    TooManyArgsForInput,
    UnresolvedIdent(istr::IStr),
}

impl std::error::Error for EncodingError {}
impl core::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

type Result<T, E = EncodingError> = std::result::Result<T, E>;

impl Encoder {
    pub fn new() -> Self {
        Self {
            keywords: keywords::Keywords::init(),
            mir: MirBuilder::new(),
            nr: name_resolver::NameResolver::new(),
            regs: mir::RegAllocator::new(),
        }
    }

    pub fn encode(&mut self, syn: &Syntax) -> Result<()> {
        let (_id, mut block) = self.mir.new_block();

        self.write_statement(syn, &mut block)?;

        dbg!(block);

        Ok(())
    }

    fn write_statement(&mut self, syn: &Syntax, bb: &mut BasicBlockBuilder) -> Result<()> {
        match self.keywords.get(syn.name) {
            Some(keywords::Keyword::Let) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.define(ident.name, &mut self.regs);
                        // bb.instrs.push(mir::Instr::StartLifetime(reg));
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [name, val] => {
                    todo!();
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForPrint),
                [_, _, _, ..] => Err(EncodingError::TooManyArgsForPrint),
            },
            Some(keywords::Keyword::Print) => match syn.args.as_slice() {
                [arg] => {
                    let val = self.write_expr(arg, bb)?;
                    bb.instrs.push(crate::mir::Instr::ConsolePrint(val));
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForPrint),
                [_, _, ..] => Err(EncodingError::TooManyArgsForPrint),
            },
            Some(keywords::Keyword::Input) => match syn.args.as_slice() {
                [ident] => {
                    if ident.args.is_empty() {
                        let reg = self.nr.resolve(ident.name)?;
                        bb.instrs.push(crate::mir::Instr::ConsoleInput(reg));
                    } else {
                        todo!()
                    }
                    Ok(())
                }
                [] => Err(EncodingError::MissingArgsForPrint),
                [_, _, ..] => Err(EncodingError::TooManyArgsForPrint),
            },
            Some(keywords::Keyword::Block) => {
                let scope = self.nr.scope();
                for arg in syn.args.as_slice() {
                    self.write_statement(arg, bb)?;
                }
                let regs = self.nr.close_scope_with(scope);
                for reg in regs {
                    // bb.instrs.push(mir::Instr::EndLifetime(reg));
                }
                Ok(())
            }
            _ => todo!(),
        }
    }

    fn write_expr(&mut self, syn: &Syntax, bb: &mut BasicBlockBuilder) -> Result<mir::Val> {
        match self.keywords.get(syn.name) {
            Some(_) => {
                todo!()
            }
            None => {
                if syn.args.is_empty() {
                    Ok(mir::Val::Reg(self.nr.resolve(syn.name)?))
                } else {
                    todo!()
                }
            }
        }
    }
}
