use core::num::NonZeroU32;

pub trait Visitor<'ast> {
    fn visit_ident(&mut self, ident: &Ident) {
        ident.default_visit(self)
    }

    fn visit_expr(&mut self, expr: &Expr<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_bin_op(&mut self, expr: &ExprBinOp<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_unary_op(&mut self, expr: &ExprUnaryOp<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_call(&mut self, expr: &ExprCall<'ast>) {
        expr.default_visit(self)
    }
}

pub trait Trivial {}

impl<T: Copy> Trivial for T {}
impl<T: Trivial> Trivial for [T] {}

pub trait Visit<'ast>: Trivial {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V);

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V);
}

impl<'ast, T: Visit<'ast>> Visit<'ast> for [T] {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        self.default_visit(v)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        for item in self {
            item.visit(v);
        }
    }
}

impl<'ast, T: ?Sized + Visit<'ast>> Visit<'ast> for &T {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        T::visit(self, v)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        T::default_visit(self, v)
    }
}

macro_rules! make_id {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(NonZeroU32);

        impl $name {
            pub const fn new(value: NonZeroU32) -> Self {
                Self(value)
            }

            pub const fn get(self) -> NonZeroU32 {
                self.0
            }
        }
    };
}

make_id!(IdentId);
#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub id: IdentId,
    pub name: istr::IBytes,
}

impl<'ast> Visit<'ast> for Ident {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        v.visit_ident(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, _v: &mut V) {}
}

make_id!(ExprId);
#[derive(Debug, Clone, Copy)]
pub struct Expr<'ast> {
    pub id: ExprId,
    pub kind: ExprKind<'ast>,
}

#[derive(Debug, Clone, Copy)]
pub enum ExprKind<'ast> {
    Ident(Ident),
    BinOp(&'ast ExprBinOp<'ast>),
    UnaryOp(&'ast ExprUnaryOp<'ast>),
    Call(&'ast ExprCall<'ast>),
}

impl<'ast> Visit<'ast> for Expr<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        v.visit_expr(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        match &self.kind {
            ExprKind::Ident(ident) => ident.visit(v),
            ExprKind::BinOp(expr) => expr.visit(v),
            ExprKind::UnaryOp(expr) => expr.visit(v),
            ExprKind::Call(expr) => expr.visit(v),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    // int op int => int
    Add,
    Sub,
    Mul,
    Div,

    // int op int = bool
    // bool op bool => bool
    CmpEq,
    CmpNe,
    CmpGt,
    CmpLt,
    CmpGe,
    CmpLe,
}

#[derive(Debug, Clone, Copy)]
pub struct ExprBinOp<'ast> {
    pub left: Expr<'ast>,
    pub op: BinOp,
    pub right: Expr<'ast>,
}

impl<'ast> Visit<'ast> for ExprBinOp<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        v.visit_expr_bin_op(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        let Self { left, right, op: _ } = self;
        left.visit(v);
        right.visit(v);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    // bool => int
    IntFromBool,
    // bool => bool
    // int => int
    Not,
    // int => int
    Neg,
}

#[derive(Debug, Clone, Copy)]
pub struct ExprUnaryOp<'ast> {
    pub op: BinOp,
    pub value: Expr<'ast>,
}

impl<'ast> Visit<'ast> for ExprUnaryOp<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        v.visit_expr_unary_op(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        let Self { value, op: _ } = self;
        value.visit(v);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExprCall<'ast> {
    pub func: Expr<'ast>,
    pub args: &'ast [Expr<'ast>],
}

impl<'ast> Visit<'ast> for ExprCall<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        v.visit_expr_call(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&self, v: &mut V) {
        let Self { func, args } = self;
        func.visit(v);
        args.visit(v)
    }
}