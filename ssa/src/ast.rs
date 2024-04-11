use core::num::NonZeroU32;
use std::num::NonZeroU16;

pub trait Visitor<'ast> {
    fn visit_ident(&mut self, ident: &'ast Ident) {
        ident.default_visit(self)
    }

    fn visit_path(&mut self, path: &'ast Path) {
        path.default_visit(self)
    }

    fn visit_let(&mut self, item_let: &'ast Let<'ast>) {
        item_let.default_visit(self)
    }

    fn visit_block(&mut self, block: &'ast Block<'ast>) {
        block.default_visit(self)
    }

    fn visit_conditional_block(&mut self, block: &'ast ConditionalBlock<'ast>) {
        block.default_visit(self)
    }

    fn visit_item_block(&mut self, block: &'ast ItemBlock<'ast>) {
        block.default_visit(self);
    }

    fn visit_conditional_item_block(&mut self, block: &'ast ConditionalItemBlock<'ast>) {
        block.default_visit(self);
    }

    fn visit_if(&mut self, item_if: &'ast If<'ast>) {
        item_if.default_visit(self)
    }

    fn visit_loop(&mut self, item_loop: &'ast Loop<'ast>) {
        item_loop.default_visit(self)
    }

    fn visit_item(&mut self, item: &'ast Item<'ast>) {
        item.default_visit(self)
    }

    fn visit_item_let(&mut self, _id: ItemId, stmt_let: &'ast Let<'ast>) {
        self.visit_let(stmt_let);
    }

    fn visit_item_if(&mut self, stmt_if: &'ast ItemIf<'ast>) {
        stmt_if.default_visit(self);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt<'ast>) {
        stmt.default_visit(self)
    }

    fn visit_stmt_let(&mut self, _id: StmtId, stmt_let: &'ast Let<'ast>) {
        self.visit_let(stmt_let);
    }

    fn visit_stmt_if(&mut self, _id: StmtId, stmt_if: &'ast If<'ast>) {
        self.visit_if(stmt_if);
    }

    fn visit_stmt_loop(&mut self, _id: StmtId, stmt_loop: &'ast Loop<'ast>) {
        self.visit_loop(stmt_loop);
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

    fn visit_expr_func(&mut self, expr: &'ast ExprFunc<'ast>) {
        expr.default_visit(self)
    }

    fn visit_func_param(&mut self, param: &'ast FuncParam<'ast>) {
        param.default_visit(self)
    }

    fn visit_expr_ident(&mut self, _id: ExprId, expr_ident: &'ast Ident) {
        self.visit_ident(expr_ident);
    }

    fn visit_expr_path(&mut self, _id: ExprId, expr_path: &'ast Path<'ast>) {
        self.visit_path(expr_path);
    }

    fn visit_expr_block(&mut self, _id: ExprId, expr_block: &'ast Block<'ast>) {
        self.visit_block(expr_block)
    }

    fn visit_expr_if(&mut self, _id: ExprId, expr_if: &'ast If<'ast>) {
        self.visit_if(expr_if);
    }

    fn visit_expr_loop(&mut self, _id: ExprId, expr_loop: &'ast Loop<'ast>) {
        self.visit_loop(expr_loop);
    }

    fn visit_expr_break(&mut self, _id: ExprId, expr: Option<&'ast Expr<'ast>>) {
        if let Some(expr) = expr {
            expr.default_visit(self)
        }
    }

    fn visit_expr_continue(&mut self, _id: ExprId, expr: Option<&'ast Expr<'ast>>) {
        if let Some(expr) = expr {
            expr.default_visit(self)
        }
    }

    fn visit_expr_return(&mut self, _id: ExprId, expr: Option<&'ast Expr<'ast>>) {
        if let Some(expr) = expr {
            expr.default_visit(self)
        }
    }

    fn visit_expr_struct(&mut self, expr_struct: &'ast ExprStruct<'ast>) {
        expr_struct.default_visit(self);
    }

    fn visit_expr_union(&mut self, expr_union: &'ast ExprUnion<'ast>) {
        expr_union.default_visit(self);
    }

    fn visit_expr_enum(&mut self, expr_enum: &'ast ExprEnum<'ast>) {
        expr_enum.default_visit(self);
    }

    fn visit_type_param(&mut self, param: &'ast TypeParam<'ast>) {
        param.default_visit(self);
    }

    fn visit_field(&mut self, field: &'ast Field<'ast>) {
        field.default_visit(self);
    }

    fn visit_type(&mut self, ty: &'ast Type<'ast>) {
        ty.default_visit(self)
    }

    fn visit_type_concrete(&mut self, ty: &'ast TypeConcrete<'ast>) {
        ty.default_visit(self)
    }

    fn visit_type_primitive(&mut self, ty: &'ast TypePrimitive) {
        ty.default_visit(self)
    }

    fn visit_type_tuple(&mut self, _id: TypeId, tys: &'ast [Type<'ast>]) {
        tys.default_visit(self)
    }

    fn visit_type_type(&mut self, _id: TypeId, _universe: u32) {}
}

pub trait Trivial {}

impl<T: Copy> Trivial for T {}
impl<T: Trivial> Trivial for [T] {}

pub trait Visit<'ast>: Trivial + serde::Serialize {
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

#[derive(Default)]
pub struct IdCtx {
    items: u32,
    exprs: u32,
    idents: u32,
    stmts: u32,
    types: u32,
}

macro_rules! make_id {
    ($name:ident $field:ident $func:ident) => {
        #[repr(transparent)]
        #[derive(Debug, serde::Serialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(NonZeroU32);

        impl $name {
            pub const fn new(value: NonZeroU32) -> Self {
                Self(value)
            }

            pub const fn from_u32(value: u32) -> Self {
                match NonZeroU32::new(value) {
                    Some(value) => Self::new(value),
                    None => panic!(concat!(
                        "You may not use a value of zero for ",
                        stringify!($name)
                    )),
                }
            }

            pub const fn get(self) -> NonZeroU32 {
                self.0
            }
        }

        impl IdCtx {
            pub fn $func(&mut self) -> $name {
                self.$field += 1;
                $name::from_u32(self.$field)
            }
        }
    };
}

fn istr_serialize<S: serde::Serializer>(value: &istr::IStr, ser: S) -> Result<S::Ok, S::Error> {
    ser.serialize_str(value.to_str())
}

make_id!(IdentId idents ident_id);
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Ident {
    pub id: IdentId,
    #[serde(serialize_with = "istr_serialize")]
    pub name: istr::IStr,
}

impl<'ast> Visit<'ast> for Ident {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_ident(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, _v: &mut V) {}
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Path<'ast> {
    pub segments: &'ast [Ident],
}

impl<'ast> Visit<'ast> for Path<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_path(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { segments } = self;
        for segment in *segments {
            segment.visit(v);
        }
    }
}

make_id!(ItemId items item_id);
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Item<'ast> {
    pub id: ItemId,
    pub kind: ItemKind<'ast>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum ItemKind<'ast> {
    If(&'ast ItemIf<'ast>),
    Let(&'ast Let<'ast>),
    Error,
}

impl<'ast> Visit<'ast> for Item<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_item(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { id, ref kind } = *self;
        match kind {
            ItemKind::If(item) => item.visit(v),
            ItemKind::Let(item) => v.visit_item_let(id, item),
            ItemKind::Error => (),
        }
    }
}

make_id!(StmtId stmts stmt_id);
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Stmt<'ast> {
    pub id: StmtId,
    pub kind: StmtKind<'ast>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum StmtKind<'ast> {
    If(&'ast If<'ast>),
    Loop(&'ast Loop<'ast>),
    Let(&'ast Let<'ast>),
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
            StmtKind::Let(stmt) => v.visit_stmt_let(id, stmt),
            StmtKind::Loop(stmt) => v.visit_stmt_loop(id, stmt),
        }
    }
}

impl StmtKind<'_> {
    pub fn has_block(&self) -> bool {
        matches!(
            self,
            StmtKind::If(_)
                | StmtKind::Loop(_)
                | StmtKind::Expr(Expr {
                    kind: ExprKind::Block(_),
                    ..
                })
        )
    }
}

make_id!(ExprId exprs expr_id);
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Expr<'ast> {
    pub id: ExprId,
    pub kind: ExprKind<'ast>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum ExprKind<'ast> {
    IntLiteral(u128),
    FloatLiteral(f64),

    Ident(&'ast Ident),
    Path(&'ast Path<'ast>),
    BinOp(&'ast ExprBinOp<'ast>),
    UnaryOp(&'ast ExprUnaryOp<'ast>),

    Call(&'ast ExprCall<'ast>),
    Func(&'ast ExprFunc<'ast>),

    Block(&'ast Block<'ast>),
    If(&'ast If<'ast>),
    Loop(&'ast Loop<'ast>),

    Break(Option<&'ast Expr<'ast>>),
    Continue(Option<&'ast Expr<'ast>>),
    Return(Option<&'ast Expr<'ast>>),

    Struct(&'ast ExprStruct<'ast>),
    Union(&'ast ExprUnion<'ast>),
    Enum(&'ast ExprEnum<'ast>),

    Error,
}

impl<'ast> Visit<'ast> for Expr<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { id, ref kind } = *self;
        match kind {
            ExprKind::IntLiteral(_) => (),
            ExprKind::FloatLiteral(_) => (),
            ExprKind::Ident(expr) => v.visit_expr_ident(id, expr),
            ExprKind::Path(expr) => v.visit_expr_path(id, expr),
            ExprKind::BinOp(expr) => expr.visit(v),
            ExprKind::UnaryOp(expr) => expr.visit(v),
            ExprKind::Call(expr) => expr.visit(v),
            ExprKind::Func(expr) => expr.visit(v),
            ExprKind::Block(expr) => v.visit_expr_block(id, expr),
            ExprKind::If(expr) => v.visit_expr_if(id, expr),
            ExprKind::Loop(expr) => v.visit_expr_loop(id, expr),
            ExprKind::Break(expr) => v.visit_expr_break(id, *expr),
            ExprKind::Continue(expr) => v.visit_expr_continue(id, *expr),
            ExprKind::Return(expr) => v.visit_expr_return(id, *expr),
            ExprKind::Struct(expr) => expr.visit(v),
            ExprKind::Union(expr) => expr.visit(v),
            ExprKind::Enum(expr) => expr.visit(v),
            ExprKind::Error => (),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum UnaryOp {
    // bool => int
    IntFromBool,
    // pointer => addr (int)
    IntFromPtr,
    // bool => bool
    // int => int
    Not,
    // int => int
    Neg,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExprUnaryOp<'ast> {
    pub op: UnaryOp,
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExprFunc<'ast> {
    pub params: &'ast [FuncParam<'ast>],
    pub body: Block<'ast>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FuncParam<'ast> {
    pub name: Ident,
    pub _lt: core::marker::PhantomData<&'ast ()>,
}

impl<'ast> Visit<'ast> for FuncParam<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_func_param(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { name, _lt: _ } = self;
        name.visit(v);
    }
}

impl<'ast> Visit<'ast> for ExprFunc<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_func(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { params, body } = self;
        params.visit(v);
        body.visit(v)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExprStruct<'ast> {
    pub params: &'ast [TypeParam<'ast>],
    pub fields: &'ast [Field<'ast>],
}

impl<'ast> Visit<'ast> for ExprStruct<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_struct(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { params, fields } = self;
        params.visit(v);
        fields.visit(v)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExprUnion<'ast> {
    pub params: &'ast [TypeParam<'ast>],
    pub variants: &'ast [Field<'ast>],
}

impl<'ast> Visit<'ast> for ExprUnion<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_union(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { params, variants } = self;
        params.visit(v);
        variants.visit(v)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExprEnum<'ast> {
    pub variants: &'ast [Ident],
}

impl<'ast> Visit<'ast> for ExprEnum<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_expr_enum(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { variants } = self;
        variants.visit(v)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct TypeParam<'ast> {
    pub name: Ident,
    pub bounds: [&'ast (); 0],
}

impl<'ast> Visit<'ast> for TypeParam<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_type_param(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { name, bounds: _ } = self;
        name.visit(v);
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Field<'ast> {
    pub name: Ident,
    pub ty: Type<'ast>,
}

impl<'ast> Visit<'ast> for Field<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_field(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { name, ty } = self;
        name.visit(v);
        ty.visit(v);
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Block<'ast> {
    pub stmts: &'ast [Stmt<'ast>],
    pub expr: Option<&'ast Expr<'ast>>,
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ItemBlock<'ast> {
    pub items: &'ast [Item<'ast>],
}

impl<'ast> Visit<'ast> for ItemBlock<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_item_block(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { items } = self;
        items.visit(v);
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ConditionalItemBlock<'ast> {
    pub cond: Expr<'ast>,
    pub block: ItemBlock<'ast>,
}

impl<'ast> Visit<'ast> for ConditionalItemBlock<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_conditional_item_block(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { cond, block } = self;
        cond.visit(v);
        block.visit(v)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ItemIf<'ast> {
    /// should have at least one block
    pub blocks: &'ast [ConditionalItemBlock<'ast>],
    pub default: Option<&'ast ItemBlock<'ast>>,
}

impl<'ast> Visit<'ast> for ItemIf<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_item_if(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { blocks, default } = self;
        for b in *blocks {
            b.visit(v);
        }
        default.visit(v);
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Let<'ast> {
    pub binding: Ident,
    pub ty: Option<Type<'ast>>,
    pub value: Option<Expr<'ast>>,
}

impl<'ast> Visit<'ast> for Let<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_let(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { binding, ty, value } = self;
        binding.visit(v);
        ty.visit(v);
        value.visit(v);
    }
}

make_id!(TypeId types type_id);
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Type<'ast> {
    pub id: TypeId,
    pub kind: TypeKind<'ast>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum TypeKind<'ast> {
    Primitive(TypePrimitive),
    Tuple(&'ast [Type<'ast>]),
    Concrete(&'ast TypeConcrete<'ast>),
    /// The type of types
    Type {
        // u32::MAX means unknown universe
        universe: u32,
    },
}

impl<'ast> Visit<'ast> for Type<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_type(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { id, ref kind } = *self;
        match kind {
            TypeKind::Primitive(ty) => ty.visit(v),
            TypeKind::Tuple(ty) => v.visit_type_tuple(id, ty),
            TypeKind::Concrete(ty) => ty.visit(v),
            TypeKind::Type { universe } => v.visit_type_type(id, *universe),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct TypeConcrete<'ast> {
    pub name: Path<'ast>,
    pub generics: &'ast [Type<'ast>],
}

impl<'ast> Visit<'ast> for TypeConcrete<'ast> {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_type_concrete(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        let Self { name, generics } = self;
        name.visit(v);
        generics.visit(v);
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum TypePrimitive {
    /// A type with one trivial constructor
    Unit,
    /// A type with no constructors
    Void,
    /// signed integers with the given number of bits
    SInt { bits: NonZeroU16 },
    /// unsigned integers with the given number of bits
    UInt { bits: NonZeroU16 },
    /// ieee-754 32-bit float
    Float32,
    /// ieee-754 64-bit float
    Float64,
    /// the address integer type, unsigned and has a target-specific number of bits
    ///
    /// has provenance, unlike normal integers
    Addr,
    /// an 8-bit scalar type which can be uninitialized
    Byte,
}

impl<'ast> Visit<'ast> for TypePrimitive {
    fn visit<V: Visitor<'ast> + ?Sized>(&'ast self, v: &mut V) {
        v.visit_type_primitive(self)
    }

    fn default_visit<V: Visitor<'ast> + ?Sized>(&'ast self, _v: &mut V) {}
}
