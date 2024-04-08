use core::num::NonZeroU32;

pub trait Visitor<'ast> {
    fn visit_ident(&mut self, ident: &'ast Ident) {
        ident.default_visit(self)
    }

    fn visit_block(&mut self, block: &'ast Block<'ast>) {
        block.default_visit(self)
    }

    fn visit_conditional_block(&mut self, block: &'ast ConditionalBlock<'ast>) {
        block.default_visit(self)
    }

    fn visit_if(&mut self, item_if: &'ast If<'ast>) {
        item_if.default_visit(self)
    }

    fn visit_loop(&mut self, item_loop: &'ast Loop<'ast>) {
        item_loop.default_visit(self)
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt<'ast>) {
        stmt.default_visit(self)
    }

    fn visit_stmt_if(&mut self, _id: StmtId, stmt_if: &'ast If<'ast>) {
        stmt_if.default_visit(self)
    }

    fn visit_stmt_loop(&mut self, _id: StmtId, stmt_loop: &'ast Loop<'ast>) {
        stmt_loop.default_visit(self)
    }

    fn visit_stmt_expr(&mut self, _id: StmtId, stmt_expr: &'ast Expr<'ast>) {
        stmt_expr.default_visit(self)
    }

    fn visit_expr(&mut self, expr: &'ast Expr<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_bin_op(&mut self, expr: &'ast ExprBinOp<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_unary_op(&mut self, expr: &'ast ExprUnaryOp<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_call(&mut self, expr: &'ast ExprCall<'ast>) {
        expr.default_visit(self)
    }

    fn visit_expr_if(&mut self, _id: ExprId, expr_if: &'ast If<'ast>) {
        expr_if.default_visit(self)
    }

    fn visit_expr_loop(&mut self, _id: ExprId, expr_loop: &'ast Loop<'ast>) {
        expr_loop.default_visit(self)
    }
}

pub trait Trivial {}

impl<T: Copy> Trivial for T {}
impl<T: Trivial> Trivial for [T] {}

pub trait Visit<'ast>: Trivial {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V);

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V);
}

impl<'ast, T: Visit<'ast>> Visit<'ast> for [T] {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        self.default_visit(v)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        for item in self {
            item.visit(v);
        }
    }
}

impl<'ast, T: Visit<'ast> + Copy> Visit<'ast> for Option<T> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        self.default_visit(v)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        if let Some(item) = self {
            item.visit(v);
        }
    }
}

impl<'ast, T: ?Sized + Visit<'ast>> Visit<'ast> for &T {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        T::visit(self, v)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
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
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_ident(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, _v: &mut V) {}
}

make_id!(StmtId);
#[derive(Debug, Clone, Copy)]
pub struct Stmt<'ast> {
    pub id: StmtId,
    pub kind: StmtKind<'ast>,
}

#[derive(Debug, Clone, Copy)]
pub enum StmtKind<'ast> {
    If(&'ast If<'ast>),
    Loop(&'ast Loop<'ast>),
    Expr(&'ast Expr<'ast>),
}

impl<'ast> Visit<'ast> for Stmt<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_stmt(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { id, ref kind } = *self;
        match kind {
            StmtKind::Expr(stmt) => v.visit_stmt_expr(id, stmt),
            StmtKind::If(stmt) => v.visit_stmt_if(id, stmt),
            StmtKind::Loop(stmt) => v.visit_stmt_loop(id, stmt),
        }
    }
}

make_id!(ExprId);
#[derive(Debug, Clone, Copy)]
pub struct Expr<'ast> {
    pub id: ExprId,
    pub kind: ExprKind<'ast>,
}

#[derive(Debug, Clone, Copy)]
pub enum ExprKind<'ast> {
    Ident(&'ast Ident),
    BinOp(&'ast ExprBinOp<'ast>),
    UnaryOp(&'ast ExprUnaryOp<'ast>),
    Call(&'ast ExprCall<'ast>),
    If(&'ast If<'ast>),
    Loop(&'ast Loop<'ast>),
}

impl<'ast> Visit<'ast> for Expr<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { id, ref kind } = *self;
        match kind {
            ExprKind::Ident(ident) => ident.visit(v),
            ExprKind::BinOp(expr) => expr.visit(v),
            ExprKind::UnaryOp(expr) => expr.visit(v),
            ExprKind::Call(expr) => expr.visit(v),
            ExprKind::If(expr) => v.visit_expr_if(id, expr),
            ExprKind::Loop(expr) => v.visit_expr_loop(id, expr),
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
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_bin_op(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
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
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_unary_op(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
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
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_call(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { func, args } = self;
        func.visit(v);
        args.visit(v)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Block<'ast> {
    pub stmts: &'ast [Stmt<'ast>],
    pub expr: Option<Expr<'ast>>,
}

impl<'ast> Visit<'ast> for Block<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_block(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { stmts, expr } = self;
        stmts.visit(v);
        expr.visit(v)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConditionalBlock<'ast> {
    pub cond: Expr<'ast>,
    pub block: Block<'ast>,
}

impl<'ast> Visit<'ast> for ConditionalBlock<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_conditional_block(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { cond, block } = self;
        cond.visit(v);
        block.visit(v)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct If<'ast> {
    /// should have at least one block
    pub blocks: &'ast [ConditionalBlock<'ast>],
    pub default: Option<&'ast Block<'ast>>,
}

impl<'ast> Visit<'ast> for If<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_if(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { blocks, default } = self;
        for b in *blocks {
            b.visit(v);
        }
        default.visit(v);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Loop<'ast> {
    pub block: Block<'ast>,
}

impl<'ast> Visit<'ast> for Loop<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_loop(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { block } = self;
        block.visit(v);
    }
}
