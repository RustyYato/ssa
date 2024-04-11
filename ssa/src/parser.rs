use std::{hash::Hash, num::NonZeroU16};

use crate::{ast, pool::Pool};

#[derive(logos::Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(source = [u8])]
#[logos(error = LexerError)]
pub enum TokenKind<'s> {
    Eof,

    // erroneous unicode characters
    #[regex(".", priority = 1)]
    UnknownChar,
    // erroneous non-utf-8 bytes
    #[regex(b".", priority = 0)]
    UnknownByte,

    // unix style newlines
    #[token("\n", priority = 10)]
    #[token("\r", priority = 10)]
    // windows style newlines
    #[token("\r\n", priority = 10)]
    Newline,
    #[regex(r"\p{White_Space}")]
    WhiteSpace,
    #[regex(r"#[^\n]*")]
    LineComment,

    #[regex(r"[_\p{XID_Start}]\p{XID_Continue}*", |lexer| {
        let ident = lexer.slice();
        #[cfg(debug_assertions)]
        core::str::from_utf8(ident).unwrap();
        // SAFETY: this regex only matches valid utf-8 characters
        unsafe { core::str::from_utf8_unchecked(ident) }
    })]
    Ident(&'s str),

    #[regex(r"[0-9]+")]
    DecimalIntegerLiteral,
    #[regex(r"0x[0-9a-fA-F]+")]
    HexIntegerLiteral,
    #[regex(r"0o[0-7]+")]
    OctalIntegerLiteral,
    #[regex(r"0b[01]+")]
    BinaryIntegerLiteral,

    #[regex(r"-?[0-9]+\.[0-9]+")]
    #[regex(r"-?[0-9]+[eE][+-]?[0-9]+")]
    #[regex(r"-?[0-9]+\.[0-9]+[eE][+-]?[0-9]+")]
    FloatLiteral,

    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("::")]
    Colon2,
    #[token(";")]
    Semicolon,
    #[token(".")]
    Dot,
    #[token("..")]
    Dot2,
    #[token("{")]
    OpenCurly,
    #[token("}")]
    CloseCurly,
    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
    #[token("[")]
    OpenSquare,
    #[token("]")]
    CloseSquare,

    #[token("=")]
    Equal,
    #[token("==")]
    Equal2,
    #[token("!=")]
    BangEqual,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,

    #[token("+")]
    Plus,
    #[token("-")]
    Hyphen,
    #[token("*")]
    Star,
    #[token("/")]
    ForwardSlash,
    #[token("%")]
    Percent,
    #[token("&")]
    Ambersand,
    #[token("&&")]
    Ambersand2,
    #[token("|")]
    Pipe,
    #[token("||")]
    Pipe2,
    #[token("^")]
    Caret,
    #[token("!")]
    Bang,
    #[token("@")]
    At,

    #[token("_", priority = 10)]
    Underscore,
    #[token("as")]
    As,
    #[token("addr")]
    Addr,
    #[token("bool")]
    Bool,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("Type")]
    Type,
    #[token("let")]
    Let,
    #[token("fn")]
    Fn,
    #[token("struct")]
    Struct,
    #[token("union")]
    Union,
    #[token("enum")]
    Enum,
}

pub trait ParseError<'text> {
    fn expected_item(&mut self, found: &[Token<'text>]);

    fn expected_ident(&mut self, found: &[Token<'text>]);

    fn expected_expr(&mut self, found: &[Token<'text>]);

    fn expected_type(&mut self, found: &[Token<'text>]);

    fn too_many_bits(&mut self, ty: char, bits: &str, tokens: &[Token<'text>]);

    fn zero_bit_integer(&mut self, is_signed: bool, tokens: &[Token<'text>]);

    fn unsupported_float_bits(&mut self, bits: u16, tokens: &[Token<'text>]);

    fn expected(&mut self, token: TokenKind, found: &[Token<'text>]);

    fn unrepresentable_int_literal(&mut self, found: &[Token<'text>]);

    fn unrepresentable_float_literal(&mut self, found: &[Token<'text>]);

    fn had_errors(self) -> HadErrors<Self>
    where
        Self: Sized,
    {
        HadErrors {
            had_errors: false,
            errors: self,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HadErrors<T = ()> {
    pub had_errors: bool,
    pub errors: T,
}

impl HadErrors {
    pub const fn new() -> Self {
        Self {
            had_errors: false,
            errors: (),
        }
    }
}

#[allow(unused)]
impl<'text> ParseError<'text> for HadErrors {
    fn expected_item(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn expected_ident(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn expected_expr(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn expected_type(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn too_many_bits(&mut self, ty: char, bits: &str, tokens: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn zero_bit_integer(&mut self, is_signed: bool, tokens: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn unsupported_float_bits(&mut self, bits: u16, tokens: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn expected(&mut self, token: TokenKind, found: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn unrepresentable_int_literal(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
    }

    fn unrepresentable_float_literal(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
    }
}

impl<'text, T: ParseError<'text>> ParseError<'text> for HadErrors<T> {
    fn expected_item(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.expected_item(found)
    }

    fn expected_ident(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.expected_ident(found)
    }

    fn expected_expr(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.expected_expr(found)
    }

    fn expected_type(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.expected_type(found)
    }

    fn too_many_bits(&mut self, ty: char, bits: &str, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.too_many_bits(ty, bits, found)
    }

    fn zero_bit_integer(&mut self, is_signed: bool, tokens: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.zero_bit_integer(is_signed, tokens)
    }

    fn unsupported_float_bits(&mut self, bits: u16, tokens: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.unsupported_float_bits(bits, tokens)
    }

    fn expected(&mut self, token: TokenKind, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.expected(token, found)
    }

    fn unrepresentable_int_literal(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.unrepresentable_int_literal(found)
    }

    fn unrepresentable_float_literal(&mut self, found: &[Token<'text>]) {
        self.had_errors = true;
        self.errors.unrepresentable_float_literal(found)
    }
}

pub struct PanicDebugParseError;

impl<'text> ParseError<'text> for PanicDebugParseError {
    fn expected_item(&mut self, found: &[Token<'text>]) {
        panic!("Expected an item, but found {:#?}", found[0])
    }

    fn expected_ident(&mut self, found: &[Token<'text>]) {
        panic!("Expected an ident, but found {:#?}", found[0])
    }

    fn expected_expr(&mut self, found: &[Token<'text>]) {
        panic!("Expected an expr, but found {:#?}", found[0])
    }

    fn expected_type(&mut self, found: &[Token<'text>]) {
        panic!("Expected a type, but found {:#?}", found[0])
    }

    fn too_many_bits(&mut self, ty: char, bits: &str, _found: &[Token<'text>]) {
        let ty = match ty {
            'i' => "signed integer",
            'u' => "unsigned integer",
            'f' => "float",
            _ => unreachable!(),
        };
        panic!(
            "Tried to specify {bits} > {} bits for a {ty}, which is not supported",
            u16::MAX
        )
    }

    fn zero_bit_integer(&mut self, is_signed: bool, _tokens: &[Token<'text>]) {
        panic!(
            "Zero bit {}signed integers are unsupported",
            if is_signed { "" } else { "un" }
        )
    }

    fn unsupported_float_bits(&mut self, bits: u16, _tokens: &[Token<'text>]) {
        panic!("{} bit floats are unsupported", bits)
    }

    fn expected(&mut self, kind: TokenKind, found: &[Token<'text>]) {
        panic!(
            "Invalid token: expected {kind:?}, but found {:?}",
            found[0].kind,
        )
    }

    fn unrepresentable_int_literal(&mut self, found: &[Token<'text>]) {
        panic!("Unrepresentable int literal: {found:?}")
    }

    fn unrepresentable_float_literal(&mut self, found: &[Token<'text>]) {
        panic!("Unrepresentable float literal: {found:?}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexerError {}

impl Default for LexerError {
    fn default() -> Self {
        extern "C-unwind" {
            fn __lexer_error_unreachable_default() -> !;
        }

        // force a linker error
        unsafe { __lexer_error_unreachable_default() }
    }
}

struct Lexer<'text> {
    logos: logos::Lexer<'text, TokenKind<'text>>,
    line: u32,
    col: u32,
}

struct PeekingLexer<'text, const N: usize> {
    peek: [Token<'text>; N],
    lexer: Lexer<'text>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct Token<'text> {
    pub kind: TokenKind<'text>,
    pub lexeme: &'text bstr::BStr,
    pub line_start: u32,
    pub col_start: u32,
    pub line_end: u32,
    pub col_end: u32,
}

pub struct Parser<'ast, 'text, 'a> {
    lexer: PeekingLexer<'text, 2>,

    errors: &'a mut dyn ParseError<'text>,

    ctx: &'ast AstContext,
    pool: ObjectPools<'ast>,
    id_ctx: ast::IdCtx,
}

#[derive(Default)]
pub struct AstContext {
    generic: bumpalo::Bump,
}

#[derive(Default)]
pub struct ObjectPools<'ast> {
    cond_item_blocks: Pool<ast::ConditionalItemBlock<'ast>, 4>,
    cond_blocks: Pool<ast::ConditionalBlock<'ast>, 4>,
    expr: Pool<ast::Expr<'ast>, 4>,
    stmt: Pool<ast::Stmt<'ast>, 16>,
    types: Pool<ast::Type<'ast>, 4>,
    item: Pool<ast::Item<'ast>, 16>,
    params: Pool<ast::TypeParam<'ast>, 4>,
    fields: Pool<ast::Field<'ast>, 8>,
    ident: Pool<ast::Ident, 4>,
}

#[derive(Clone, Copy)]
enum ExprOp {
    BinOp(ast::BinOp),
    UnaryOp(ast::UnaryOp),
    Call,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum ExprPrec {
    Expr,
    Sum,
    Product,
    Postfix,
}

impl AstContext {
    pub fn reset(&mut self) {
        self.generic.reset();
    }

    fn alloc<T: Copy>(&self, x: T) -> &T {
        self.generic.alloc(x)
    }

    fn alloc_slice<T: Copy>(&self, x: &[T]) -> &[T] {
        self.generic.alloc_slice_copy(x)
    }
}

impl ObjectPools<'_> {
    pub fn clear<'a>(self) -> ObjectPools<'a> {
        ObjectPools {
            cond_item_blocks: self.cond_item_blocks.reuse(),
            cond_blocks: self.cond_blocks.reuse(),
            expr: self.expr.reuse(),
            stmt: self.stmt.reuse(),
            types: self.types.reuse(),
            ident: self.ident.reuse(),
            item: self.item.reuse(),
            fields: self.fields.reuse(),
            params: self.params.reuse(),
        }
    }
}

impl<'text> Lexer<'text> {
    fn new(text: &'text [u8]) -> Self {
        Self {
            logos: logos::Lexer::new(text),
            line: 0,
            col: 0,
        }
    }

    fn next_token(&mut self) -> Token<'text> {
        loop {
            let line_start = self.line;
            let col_start = self.col;
            let kind = self.raw_next_token_kind();
            let lexeme = self.logos.slice();
            if let TokenKind::Newline = kind {
                self.line += 1;
                self.col = 0;
            } else {
                self.col += lexeme.len() as u32;
            }

            let line_end = self.line;
            let col_end = self.col;

            if !matches!(
                kind,
                TokenKind::Newline | TokenKind::WhiteSpace | TokenKind::LineComment
            ) {
                return Token {
                    kind,
                    lexeme: bstr::BStr::new(lexeme),
                    line_start,
                    col_start,
                    line_end,
                    col_end,
                };
            }
        }
    }

    fn raw_next_token_kind(&mut self) -> TokenKind<'text> {
        self.logos
            .next()
            .unwrap_or(Ok(TokenKind::Eof))
            .unwrap_or_else(|inf| match inf {})
    }
}

impl<'text, const N: usize> PeekingLexer<'text, N> {
    fn new(text: &'text [u8]) -> Self {
        let mut lexer = Lexer::new(text);
        Self {
            peek: core::array::from_fn(|_| lexer.next_token()),
            lexer,
        }
    }

    fn next_token(&mut self) -> Token<'text> {
        let token = self.peek[0];
        self.skip_tokens::<1>();
        token
    }

    fn skip_tokens<const M: usize>(&mut self) {
        self.peek.copy_within(M.., 0);
        for token in self.peek[N - M..].iter_mut() {
            *token = self.lexer.next_token();
        }
    }
}

impl<'ast, 'text, 'env> Parser<'ast, 'text, 'env> {
    pub fn new(
        ctx: &'ast AstContext,
        pool: ObjectPools<'ast>,
        errors: &'env mut dyn ParseError<'text>,
        text: &'text [u8],
    ) -> Self {
        Self {
            ctx,
            errors,
            lexer: PeekingLexer::new(text),
            id_ctx: ast::IdCtx::default(),
            pool,
        }
    }

    pub fn into_pool(self) -> ObjectPools<'static> {
        self.pool.clear()
    }

    pub fn clear_text<'a, 'e>(self, errors: &'e mut dyn ParseError<'a>) -> Parser<'ast, 'a, 'e> {
        Parser {
            ctx: self.ctx,
            errors,
            lexer: PeekingLexer::new(b""),
            id_ctx: ast::IdCtx::default(),
            pool: self.pool,
        }
    }

    pub fn set_text(&mut self, text: &'text [u8]) {
        self.id_ctx = ast::IdCtx::default();
        self.lexer = PeekingLexer::new(text);
    }

    fn parse(&mut self, kind: TokenKind) -> bool {
        if self.lexer.peek[0].kind == kind {
            self.lexer.next_token();
            true
        } else {
            false
        }
    }

    fn debug_expect(&mut self, kind: TokenKind) -> Token<'text> {
        debug_assert_eq!(self.lexer.peek[0].kind, kind);
        self.lexer.next_token()
    }

    fn expect(&mut self, kind: TokenKind) {
        if self.lexer.peek[0].kind == kind {
            self.lexer.next_token();
        } else {
            self.errors.expected(kind, &self.lexer.peek);
        }
    }

    fn peek(&self) -> TokenKind<'text> {
        self.lexer.peek[0].kind
    }

    pub fn parse_file(&mut self) -> &'ast [ast::Item<'ast>] {
        let mut items = self.pool.item.alloc();

        while self.peek() != TokenKind::Eof {
            items.push(self.parse_item());
        }

        let file = self.ctx.alloc_slice(&items);

        self.pool.item.free(items);

        self.expect(TokenKind::Eof);

        file
    }

    fn parse_item(&mut self) -> ast::Item<'ast> {
        let kind = match self.peek() {
            TokenKind::If => ast::ItemKind::If(self.parse_item_if()),
            TokenKind::Let => {
                let item = ast::ItemKind::Let(self.parse_let());
                self.expect(TokenKind::Semicolon);
                item
            }
            _ => {
                self.lexer.next_token();
                self.errors.expected_item(&self.lexer.peek);
                ast::ItemKind::Error
            }
        };

        ast::Item {
            id: self.id_ctx.item_id(),
            kind,
        }
    }

    fn parse_stmt(&mut self) -> ast::Stmt<'ast> {
        let kind = match self.peek() {
            TokenKind::If => ast::StmtKind::If(self.parse_if()),
            TokenKind::Let => ast::StmtKind::Let(self.parse_let()),
            TokenKind::Loop => ast::StmtKind::Loop(self.parse_loop()),
            _ => ast::StmtKind::Expr(self.ctx.alloc(self.parse_expr())),
        };

        ast::Stmt {
            id: self.id_ctx.stmt_id(),
            kind,
        }
    }

    fn parse_ident(&mut self) -> ast::Ident {
        match self.peek() {
            TokenKind::Ident(value) => {
                self.lexer.next_token();
                ast::Ident {
                    id: self.id_ctx.ident_id(),
                    name: istr::IStr::new(value),
                }
            }
            TokenKind::Addr => ast::Ident {
                id: self.id_ctx.ident_id(),
                name: istr::IStr::new("addr"),
            },
            _ => {
                self.errors.expected_ident(&self.lexer.peek);
                ast::Ident {
                    id: self.id_ctx.ident_id(),
                    name: istr::IStr::empty(),
                }
            }
        }
    }

    fn parse_path(&mut self, first: Option<ast::Ident>) -> ast::Path<'ast> {
        let mut pool_idents = self.pool.ident.alloc();
        pool_idents.push(first.unwrap_or_else(|| self.parse_ident()));

        while self.parse(TokenKind::Colon2) {
            pool_idents.push(self.parse_ident())
        }

        let segments = self.ctx.alloc_slice(&pool_idents);
        self.pool.ident.free(pool_idents);

        ast::Path { segments }
    }

    fn parse_item_if(&mut self) -> &'ast ast::ItemIf<'ast> {
        self.debug_expect(TokenKind::If);
        let cond = self.parse_expr();
        let block = self.parse_item_block();
        let mut blocks = self.pool.cond_item_blocks.alloc();
        blocks.push(ast::ConditionalItemBlock { cond, block });
        let mut default = None;

        while self.parse(TokenKind::Else) {
            if self.parse(TokenKind::If) {
                let cond = self.parse_expr();
                let block = self.parse_item_block();

                blocks.push(ast::ConditionalItemBlock { cond, block });
            } else {
                default = Some(self.ctx.alloc(self.parse_item_block()));
                break;
            }
        }

        let cond_blocks = self.ctx.alloc_slice(&blocks);

        self.pool.cond_item_blocks.free(blocks);

        self.ctx.alloc(ast::ItemIf {
            blocks: cond_blocks,
            default,
        })
    }

    fn parse_if(&mut self) -> &'ast ast::If<'ast> {
        self.debug_expect(TokenKind::If);
        let cond = self.parse_expr();
        let block = self.parse_block();
        let mut blocks = self.pool.cond_blocks.alloc();
        blocks.push(ast::ConditionalBlock { cond, block });
        let mut default = None;

        while self.parse(TokenKind::Else) {
            if self.parse(TokenKind::If) {
                let cond = self.parse_expr();
                let block = self.parse_block();

                blocks.push(ast::ConditionalBlock { cond, block });
            } else {
                default = Some(self.ctx.alloc(self.parse_block()));
                break;
            }
        }

        let cond_blocks = self.ctx.alloc_slice(&blocks);

        self.pool.cond_blocks.free(blocks);

        self.ctx.alloc(ast::If {
            blocks: cond_blocks,
            default,
        })
    }

    fn parse_loop(&mut self) -> &'ast ast::Loop<'ast> {
        self.debug_expect(TokenKind::Loop);
        let block = self.parse_block();
        self.ctx.alloc(ast::Loop { block })
    }

    fn parse_let(&mut self) -> &'ast ast::Let<'ast> {
        self.debug_expect(TokenKind::Let);
        let name = self.parse_ident();
        let ty = if self.parse(TokenKind::Colon) {
            Some(self.parse_type())
        } else {
            None
        };
        let expr = if self.parse(TokenKind::Equal) {
            Some(self.parse_expr())
        } else {
            None
        };
        self.ctx.alloc(ast::Let {
            binding: name,
            ty,
            value: expr,
        })
    }

    fn parse_expr(&mut self) -> ast::Expr<'ast> {
        self.parse_expr_in(ExprPrec::Expr)
    }

    fn parse_expr_in(&mut self, prec: ExprPrec) -> ast::Expr<'ast> {
        let mut expr = self.parse_expr_terminal(prec);

        while let Some((op, left, right)) = self.peek_expr_postfix() {
            if prec >= left {
                break;
            }

            expr = self.finish_expr_postfix(expr, op, right)
        }

        expr
    }

    fn is_expr_start(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident(_)
                | TokenKind::Addr
                | TokenKind::OpenCurly
                | TokenKind::If
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Return
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
        )
    }

    fn parse_expr_terminal(&mut self, prec: ExprPrec) -> ast::Expr<'ast> {
        let kind = match self.peek() {
            TokenKind::Ident(_) | TokenKind::Addr => {
                ast::ExprKind::Ident(self.ctx.alloc(self.parse_ident()))
            }
            TokenKind::OpenCurly => ast::ExprKind::Block(self.ctx.alloc(self.parse_block())),
            TokenKind::If => ast::ExprKind::If(self.parse_if()),
            TokenKind::Loop => ast::ExprKind::Loop(self.parse_loop()),
            TokenKind::Break => {
                self.debug_expect(TokenKind::Break);
                ast::ExprKind::Break(if self.is_expr_start() {
                    Some(self.ctx.alloc(self.parse_expr_in(prec)))
                } else {
                    None
                })
            }
            TokenKind::Continue => {
                self.debug_expect(TokenKind::Continue);
                ast::ExprKind::Continue(if self.is_expr_start() {
                    Some(self.ctx.alloc(self.parse_expr_in(prec)))
                } else {
                    None
                })
            }
            TokenKind::Return => {
                self.debug_expect(TokenKind::Return);
                ast::ExprKind::Return(if self.is_expr_start() {
                    Some(self.ctx.alloc(self.parse_expr_in(prec)))
                } else {
                    None
                })
            }
            TokenKind::Struct => {
                self.debug_expect(TokenKind::Struct);
                let params = self.parse_type_params();
                self.expect(TokenKind::OpenCurly);
                let fields = self.parse_comma_seperated_until(
                    TokenKind::CloseCurly,
                    |pool| &mut pool.fields,
                    Self::parse_field_definition,
                );

                ast::ExprKind::Struct(self.ctx.alloc(ast::ExprStruct { params, fields }))
            }
            TokenKind::Union => {
                self.debug_expect(TokenKind::Union);
                let params = self.parse_type_params();
                self.expect(TokenKind::OpenCurly);
                let variants = self.parse_comma_seperated_until(
                    TokenKind::CloseCurly,
                    |pool| &mut pool.fields,
                    Self::parse_field_definition,
                );

                ast::ExprKind::Union(self.ctx.alloc(ast::ExprUnion { params, variants }))
            }
            TokenKind::Enum => {
                self.debug_expect(TokenKind::Enum);
                self.expect(TokenKind::OpenCurly);
                let variants = self.parse_comma_seperated_until(
                    TokenKind::CloseCurly,
                    |pool| &mut pool.ident,
                    Self::parse_ident,
                );

                ast::ExprKind::Enum(self.ctx.alloc(ast::ExprEnum { variants }))
            }
            TokenKind::DecimalIntegerLiteral => {
                let peek = self.lexer.peek;
                let token = self.debug_expect(TokenKind::DecimalIntegerLiteral);
                let lexeme: &[u8] = token.lexeme.as_ref();
                let lexeme = unsafe { core::str::from_utf8_unchecked(lexeme) };

                ast::ExprKind::IntLiteral(match lexeme.parse() {
                    Ok(value) => value,
                    Err(_) => {
                        self.errors.unrepresentable_int_literal(&peek);
                        0
                    }
                })
            }
            TokenKind::HexIntegerLiteral => {
                let peek = self.lexer.peek;
                let token = self.debug_expect(TokenKind::HexIntegerLiteral);
                let lexeme: &[u8] = token.lexeme.as_ref();
                let lexeme = unsafe { core::str::from_utf8_unchecked(&lexeme[2..]) };

                ast::ExprKind::IntLiteral(match u128::from_str_radix(lexeme, 16) {
                    Ok(value) => value,
                    Err(_) => {
                        self.errors.unrepresentable_int_literal(&peek);
                        0
                    }
                })
            }
            TokenKind::OctalIntegerLiteral => {
                let peek = self.lexer.peek;
                let token = self.debug_expect(TokenKind::OctalIntegerLiteral);
                let lexeme: &[u8] = token.lexeme.as_ref();
                let lexeme = unsafe { core::str::from_utf8_unchecked(&lexeme[2..]) };

                ast::ExprKind::IntLiteral(match u128::from_str_radix(lexeme, 8) {
                    Ok(value) => value,
                    Err(_) => {
                        self.errors.unrepresentable_int_literal(&peek);
                        0
                    }
                })
            }
            TokenKind::BinaryIntegerLiteral => {
                let peek = self.lexer.peek;
                let token = self.debug_expect(TokenKind::BinaryIntegerLiteral);
                let lexeme: &[u8] = token.lexeme.as_ref();
                let lexeme = unsafe { core::str::from_utf8_unchecked(&lexeme[2..]) };

                ast::ExprKind::IntLiteral(match u128::from_str_radix(lexeme, 2) {
                    Ok(value) => value,
                    Err(_) => {
                        self.errors.unrepresentable_int_literal(&peek);
                        0
                    }
                })
            }
            TokenKind::FloatLiteral => {
                let peek = self.lexer.peek;
                let token = self.debug_expect(TokenKind::FloatLiteral);
                let lexeme: &[u8] = token.lexeme.as_ref();
                let lexeme = unsafe { core::str::from_utf8_unchecked(lexeme) };

                ast::ExprKind::FloatLiteral(match lexeme.parse() {
                    Ok(value) => value,
                    Err(_) => {
                        self.errors.unrepresentable_float_literal(&peek);
                        f64::NAN
                    }
                })
            }
            _ => {
                debug_assert!(!self.is_expr_start());
                self.errors.expected_expr(&self.lexer.peek);
                ast::ExprKind::Error
            }
        };

        ast::Expr {
            id: self.id_ctx.expr_id(),
            kind,
        }
    }

    fn parse_type_params(&mut self) -> &'ast [ast::TypeParam<'ast>] {
        if self.parse(TokenKind::OpenSquare) {
            self.parse_comma_seperated_until(
                TokenKind::CloseSquare,
                |pool| &mut pool.params,
                Self::parse_type_param,
            )
        } else {
            &[]
        }
    }

    fn parse_type_param(&mut self) -> ast::TypeParam<'ast> {
        let name = self.parse_ident();
        // TODO: implemnt  bounds
        ast::TypeParam { name, bounds: [] }
    }

    fn parse_field_definition(&mut self) -> ast::Field<'ast> {
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);
        let ty = self.parse_type();
        ast::Field { name, ty }
    }

    fn parse_comma_seperated_until<T: Copy, const N: usize>(
        &mut self,
        end: TokenKind,
        pool: impl for<'a> Fn(&'a mut ObjectPools<'ast>) -> &'a mut Pool<T, N>,
        mut item: impl FnMut(&mut Self) -> T,
    ) -> &'ast [T] {
        let mut args = pool(&mut self.pool).alloc();
        loop {
            match self.peek() {
                TokenKind::Eof => break,
                tok if tok == end => break,
                _ => (),
            }

            args.push(item(self));

            if !self.parse(TokenKind::Comma) {
                break;
            }
        }
        self.expect(end);

        let args_list = self.ctx.alloc_slice(&args);

        pool(&mut self.pool).free(args);

        args_list
    }

    fn parse_args_list(&mut self) -> &'ast [ast::Expr<'ast>] {
        self.expect(TokenKind::OpenParen);
        self.parse_comma_seperated_until(
            TokenKind::CloseParen,
            |pool| &mut pool.expr,
            Self::parse_expr,
        )
    }

    fn finish_expr_postfix(
        &mut self,
        expr: ast::Expr<'ast>,
        op: ExprOp,
        right: ExprPrec,
    ) -> ast::Expr<'ast> {
        let kind = match op {
            ExprOp::BinOp(op) => {
                self.lexer.skip_tokens::<1>();
                let right = self.parse_expr_in(right);
                ast::ExprKind::BinOp(self.ctx.alloc(ast::ExprBinOp {
                    left: expr,
                    op,
                    right,
                }))
            }
            ExprOp::UnaryOp(op) => {
                match op {
                    ast::UnaryOp::IntFromBool => todo!(),
                    ast::UnaryOp::IntFromPtr => self.lexer.skip_tokens::<2>(),
                    ast::UnaryOp::Not | ast::UnaryOp::Neg => self.lexer.skip_tokens::<1>(),
                }

                ast::ExprKind::UnaryOp(self.ctx.alloc(ast::ExprUnaryOp { op, value: expr }))
            }
            ExprOp::Call => {
                let args = self.parse_args_list();
                ast::ExprKind::Call(self.ctx.alloc(ast::ExprCall { func: expr, args }))
            }
        };

        ast::Expr {
            id: self.id_ctx.expr_id(),
            kind,
        }
    }

    fn peek_expr_postfix(&self) -> Option<(ExprOp, ExprPrec, ExprPrec)> {
        let [a, b, ..] = self.lexer.peek;
        Some(match [a.kind, b.kind] {
            [TokenKind::Plus, _] => (ExprOp::BinOp(ast::BinOp::Add), ExprPrec::Sum, ExprPrec::Sum),
            [TokenKind::Hyphen, _] => {
                (ExprOp::BinOp(ast::BinOp::Sub), ExprPrec::Sum, ExprPrec::Sum)
            }
            [TokenKind::Star, _] => (
                ExprOp::BinOp(ast::BinOp::Mul),
                ExprPrec::Product,
                ExprPrec::Product,
            ),
            [TokenKind::ForwardSlash, _] => (
                ExprOp::BinOp(ast::BinOp::Div),
                ExprPrec::Product,
                ExprPrec::Product,
            ),
            [TokenKind::As, TokenKind::Addr] => (
                ExprOp::UnaryOp(ast::UnaryOp::IntFromPtr),
                ExprPrec::Postfix,
                ExprPrec::Postfix,
            ),
            [TokenKind::OpenParen, _] => (ExprOp::Call, ExprPrec::Postfix, ExprPrec::Postfix),
            _ => return None,
        })
    }

    fn parse_item_block(&mut self) -> ast::ItemBlock<'ast> {
        let mut pool_items = self.pool.item.alloc();
        self.expect(TokenKind::OpenCurly);
        while !matches!(self.peek(), TokenKind::CloseCurly | TokenKind::Eof) {
            pool_items.push(self.parse_item());
        }
        self.expect(TokenKind::CloseCurly);
        let items = self.ctx.alloc_slice(&pool_items);
        self.pool.item.free(pool_items);

        ast::ItemBlock { items }
    }

    fn parse_block(&mut self) -> ast::Block<'ast> {
        let mut pool_stmts = self.pool.stmt.alloc();
        self.expect(TokenKind::OpenCurly);
        let mut last = None::<ast::Stmt>;
        loop {
            if let Some(stmt) = last {
                if !stmt.kind.has_block() && !self.parse(TokenKind::Semicolon) {
                    break;
                }
                pool_stmts.push(stmt);
                last = None;
            }

            let stmt = match self.peek() {
                TokenKind::CloseCurly | TokenKind::Eof => break,
                TokenKind::Semicolon => {
                    self.lexer.next_token();
                    continue;
                }
                _ => self.parse_stmt(),
            };
            last = Some(stmt);
        }
        self.expect(TokenKind::CloseCurly);
        let mut expr = None;
        if let Some(ast::Stmt {
            kind: ast::StmtKind::Expr(e),
            ..
        }) = last
        {
            expr = Some(e);
        } else if let Some(last) = last {
            pool_stmts.push(last);

            if !last.kind.has_block() {
                self.expect(TokenKind::Semicolon);
            }
        }

        let stmts = self.ctx.alloc_slice(&pool_stmts);
        self.pool.stmt.free(pool_stmts);
        ast::Block { stmts, expr }
    }

    fn parse_type(&mut self) -> ast::Type<'ast> {
        let kind = match self.peek() {
            TokenKind::Ident(ident) => {
                if ident.starts_with(['i', 'u', 'f'])
                    && ident[1..].as_bytes().iter().all(|x| x.is_ascii_digit())
                {
                    let bits = match ident[1..].parse::<u16>() {
                        Ok(x) => x,
                        Err(_) => {
                            self.errors.too_many_bits(
                                ident.as_bytes()[0] as char,
                                &ident[1..],
                                &self.lexer.peek,
                            );
                            1
                        }
                    };

                    let kind = match (ident.as_bytes()[0], bits) {
                        (b'i', 0) => {
                            self.errors.zero_bit_integer(true, &self.lexer.peek);
                            ast::TypePrimitive::SInt {
                                bits: NonZeroU16::new(1).unwrap(),
                            }
                        }
                        (b'u', 0) => {
                            self.errors.zero_bit_integer(false, &self.lexer.peek);
                            ast::TypePrimitive::UInt {
                                bits: NonZeroU16::new(1).unwrap(),
                            }
                        }
                        (b'f', 32) => ast::TypePrimitive::Float32,
                        (b'f', 64) => ast::TypePrimitive::Float64,
                        (b'i', bits) => {
                            let bits = NonZeroU16::new(bits).unwrap();
                            ast::TypePrimitive::SInt { bits }
                        }
                        (b'u', bits) => {
                            let bits = NonZeroU16::new(bits).unwrap();
                            ast::TypePrimitive::UInt { bits }
                        }
                        (b'f', _) => {
                            self.errors.unsupported_float_bits(bits, &self.lexer.peek);
                            ast::TypePrimitive::Float32
                        }
                        _ => unreachable!(),
                    };
                    self.lexer.next_token();

                    ast::TypeKind::Primitive(kind)
                } else if ident == "byte" {
                    self.lexer.next_token();
                    ast::TypeKind::Primitive(ast::TypePrimitive::Byte)
                } else if ident == "void" {
                    self.lexer.next_token();
                    ast::TypeKind::Primitive(ast::TypePrimitive::Void)
                } else {
                    ast::TypeKind::Concrete(self.parse_type_concrete())
                }
            }
            TokenKind::OpenParen => {
                self.debug_expect(TokenKind::OpenParen);
                let tys = self.parse_type_list(TokenKind::CloseParen);
                if tys.is_empty() {
                    ast::TypeKind::Primitive(ast::TypePrimitive::Unit)
                } else {
                    ast::TypeKind::Tuple(tys)
                }
            }
            TokenKind::Type => {
                self.debug_expect(TokenKind::Type);
                ast::TypeKind::Type { universe: u32::MAX }
            }
            TokenKind::Addr => {
                self.lexer.next_token();
                ast::TypeKind::Primitive(ast::TypePrimitive::Addr)
            }
            _ => {
                self.errors.expected_type(&self.lexer.peek);
                ast::TypeKind::Primitive(ast::TypePrimitive::Void)
            }
        };

        ast::Type {
            id: self.id_ctx.type_id(),
            kind,
        }
    }

    fn parse_generics(&mut self) -> &'ast [ast::Type<'ast>] {
        if self.parse(TokenKind::OpenSquare) {
            self.parse_type_list(TokenKind::CloseSquare)
        } else {
            &[]
        }
    }

    fn parse_type_list(&mut self, end: TokenKind) -> &'ast [ast::Type<'ast>] {
        self.parse_comma_seperated_until(end, |pool| &mut pool.types, Self::parse_type)
    }

    fn parse_type_concrete(&mut self) -> &'ast ast::TypeConcrete<'ast> {
        let name = self.parse_path(None);
        let generics = self.parse_generics();
        self.ctx.alloc(ast::TypeConcrete { name, generics })
    }
}

#[test]
fn test_invalid_utf8() {
    // in release mode this should not trigger a linker error
    let lexer = logos::Lexer::<TokenKind>::new(b"\xf0\x9f\x98");
    assert!(lexer.eq([Ok(TokenKind::UnknownByte); 3]));
}

#[test]
fn test_if() {
    let lexer = logos::Lexer::<TokenKind>::new(b"else if");
    assert!(lexer.eq([
        Ok(TokenKind::Else),
        Ok(TokenKind::WhiteSpace),
        Ok(TokenKind::If)
    ]));
    let lexer = logos::Lexer::<TokenKind>::new(b"elseif");
    assert!(lexer.eq([Ok(TokenKind::Ident("elseif")); 1]));
}
