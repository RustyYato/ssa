use std::marker::PhantomData;

use crate::mir::{Reg, RegAllocator};

use super::EncodingError;

pub(crate) struct NameResolver {
    names: istr::IStrMap<Vec<Reg>>,
    scope: Vec<istr::IStr>,
}

pub struct ScopeToken(usize);
#[derive(Clone, Copy)]
pub struct ScopeRef<'a>(usize, PhantomData<&'a ScopeToken>);

impl core::fmt::Debug for ScopeRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScopeRef({})", self.0)
    }
}

trait ConstMaybe {
    type Val<T>;

    fn new<T>(f: impl FnOnce() -> T) -> Self::Val<T>;

    fn mutate<T>(val: &mut Self::Val<T>, f: impl FnOnce(&mut T));
}

struct Just;
struct Nothing;

impl ConstMaybe for Just {
    type Val<T> = T;

    fn new<T>(f: impl FnOnce() -> T) -> Self::Val<T> {
        f()
    }

    fn mutate<T>(val: &mut Self::Val<T>, f: impl FnOnce(&mut T)) {
        f(val)
    }
}

impl ConstMaybe for Nothing {
    type Val<T> = ();

    fn new<T>(_f: impl FnOnce() -> T) -> Self::Val<T> {}

    fn mutate<T>(_val: &mut Self::Val<T>, _f: impl FnOnce(&mut T)) {}
}

impl NameResolver {
    pub fn new() -> Self {
        Self {
            names: istr::IStrMap::default(),
            scope: Vec::new(),
        }
    }

    pub fn define(&mut self, name: istr::IStr, regs: &mut RegAllocator) -> Reg {
        let reg = regs.create();
        self.names.entry(name).or_default().push(reg);
        self.scope.push(name);
        reg
    }

    pub fn resolve(&self, name: istr::IStr) -> Result<Reg, EncodingError> {
        self.names
            .get(&name)
            .ok_or(EncodingError::UnresolvedIdent(name))?
            .last()
            .copied()
            .ok_or(EncodingError::UnresolvedIdent(name))
    }

    pub fn scope(&self) -> ScopeToken {
        ScopeToken(self.scope.len())
    }

    pub fn close_scope_with(&mut self, token: ScopeToken) -> Vec<Reg> {
        let mut regs = Vec::new();
        for ident in self.scope.drain(token.0..).rev() {
            regs.push(self.names.get_mut(&ident).unwrap().pop().unwrap());
        }
        regs
    }

    pub fn scope_bindings<'a>(&mut self, token: impl Into<ScopeRef<'a>>) -> Vec<Reg> {
        self.killed_nodes_(token.into())
    }

    fn killed_nodes_(&mut self, token: ScopeRef<'_>) -> Vec<Reg> {
        let mut regs = Vec::new();
        for ident in self.scope[token.0..].iter().rev() {
            regs.push(self.names.get_mut(ident).unwrap().pop().unwrap());
        }
        for (ident, reg) in self.scope[token.0..].iter().rev().zip(&regs) {
            self.names.get_mut(ident).unwrap().push(*reg);
        }
        regs
    }
}

impl ScopeToken {
    #[inline]
    pub fn as_ref(&self) -> ScopeRef<'_> {
        ScopeRef(self.0, PhantomData)
    }
}

impl<'a> From<&'a ScopeToken> for ScopeRef<'a> {
    #[inline]
    fn from(value: &'a ScopeToken) -> Self {
        value.as_ref()
    }
}
