use crate::{ast, pool::Pool};

#[derive(logos::Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(source = [u8])]
#[logos(error = LexerError)]
enum TokenKind<'s> {
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

    #[regex(r"[_\p{XID_Start}]\p{XID_Continue}*", |lexer| {
        let ident = lexer.slice();
        #[cfg(debug_assertions)]
        core::str::from_utf8(ident).unwrap();
        // SAFETY: this regex only matches valid utf-8 characters
        unsafe { core::str::from_utf8_unchecked(ident) }
    })]
    Ident(&'s str),

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
    #[token("#")]
    Pound,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LexerError {}

impl Default for LexerError {
    fn default() -> Self {
        extern "C" {
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

#[derive(Clone, Copy)]
struct Token<'text> {
    kind: TokenKind<'text>,
    lexeme: &'text [u8],
    line_start: u32,
    col_start: u32,
    line_end: u32,
    col_end: u32,
}

pub struct Parser<'ast, 'text> {
    lexer: PeekingLexer<'text, 2>,

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
    expr: Pool<ast::Expr<'ast>, 4>,
    item: Pool<ast::Item<'ast>, 16>,
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
    fn alloc<T: Copy>(&self, x: T) -> &T {
        self.generic.alloc(x)
    }

    fn alloc_slice<T: Copy>(&self, x: &[T]) -> &[T] {
        self.generic.alloc_slice_copy(x)
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

            if !matches!(kind, TokenKind::Newline | TokenKind::WhiteSpace) {
                return Token {
                    kind,
                    lexeme,
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

impl<'ast, 'text> Parser<'ast, 'text> {
    pub fn new(ctx: &'ast AstContext, pool: ObjectPools<'ast>, text: &'text [u8]) -> Self {
        Self {
            ctx,
            lexer: PeekingLexer::new(text),
            id_ctx: ast::IdCtx::default(),
            pool,
        }
    }

    pub fn into_pool(self) -> ObjectPools<'static> {
        ObjectPools {
            cond_item_blocks: self.pool.cond_item_blocks.reuse(),
            expr: self.pool.expr.reuse(),
            item: self.pool.item.reuse(),
        }
    }

    pub fn clear_text<'a>(self) -> Parser<'ast, 'a> {
        Parser {
            ctx: self.ctx,
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

    fn debug_expect(&mut self, kind: TokenKind) {
        debug_assert_eq!(self.lexer.peek[0].kind, kind);
        self.lexer.next_token();
    }

    fn expect(&mut self, kind: TokenKind) {
        if self.lexer.peek[0].kind == kind {
            self.lexer.next_token();
        } else {
            panic!(
                "Invalid token: expected {kind:?}, but found {:?}",
                self.lexer.peek[0].kind,
            )
        }
    }

    fn peek(&self) -> TokenKind<'text> {
        self.lexer.peek[0].kind
    }

    pub fn parse_file(&mut self) -> &'ast [ast::Item<'ast>] {
        let mut items = self.pool.item.alloc();

        loop {
            let kind = match self.peek() {
                TokenKind::Eof => break,
                TokenKind::If => ast::ItemKind::If(self.parse_item_if()),
                TokenKind::Let => {
                    let item = ast::ItemKind::Let(self.parse_let());
                    self.expect(TokenKind::Semicolon);
                    item
                }
                _ => unreachable!(),
            };

            items.push(ast::Item {
                id: self.id_ctx.item_id(),
                kind,
            });
        }

        let file = self.ctx.alloc_slice(&items);

        self.pool.item.free(items);

        file
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
            _ => unreachable!(),
        }
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

    fn parse_let(&mut self) -> &'ast ast::Let<'ast> {
        self.debug_expect(TokenKind::Let);
        let name = self.parse_ident();
        let expr = if self.parse(TokenKind::Equal) {
            Some(self.parse_expr())
        } else {
            None
        };
        self.ctx.alloc(ast::Let {
            binding: name,
            ty: None,
            value: expr,
        })
    }

    fn parse_expr(&mut self) -> ast::Expr<'ast> {
        self.parse_expr_in(ExprPrec::Expr)
    }

    fn parse_expr_in(&mut self, prec: ExprPrec) -> ast::Expr<'ast> {
        let mut expr = self.parse_expr_terminal();

        while let Some((op, left, right)) = self.peek_expr_postfix() {
            if prec >= left {
                break;
            }

            expr = self.finish_expr_postfix(expr, op, right)
        }

        expr
    }

    fn parse_expr_terminal(&mut self) -> ast::Expr<'ast> {
        let kind = match self.peek() {
            TokenKind::Ident(_) => ast::ExprKind::Ident(self.ctx.alloc(self.parse_ident())),
            _ => unreachable!(),
        };

        ast::Expr {
            id: self.id_ctx.expr_id(),
            kind,
        }
    }

    fn parse_args_list(&mut self) -> &'ast [ast::Expr<'ast>] {
        let mut args = self.pool.expr.alloc();
        self.expect(TokenKind::OpenParen);
        loop {
            match self.peek() {
                TokenKind::CloseParen | TokenKind::Eof => break,
                _ => (),
            }

            args.push(self.parse_expr());

            if !self.parse(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::CloseParen);

        let args_list = self.ctx.alloc_slice(&args);

        self.pool.expr.free(args);

        args_list
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
        todo!()
    }

    fn parse_block(&mut self) -> ast::Block<'ast> {
        todo!()
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

#[test]
fn test_expr() {
    let ctx = AstContext::default();
    let mut parser = Parser::new(&ctx, ObjectPools::default(), b"let x = ast + b + x;");
    let expr = parser.parse_file();
    dbg!(expr);
    panic!()
}
