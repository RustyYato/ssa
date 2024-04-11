use crate::{ast, parser};

#[derive(Clone, Copy, yoke::Yokeable)]
pub struct FileRef<'ast> {
    pub items: &'ast [ast::Item<'ast>],
}

struct AstContextHolder {
    ctx: parser::AstContext,
}

impl core::ops::Deref for AstContextHolder {
    type Target = parser::AstContext;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

// This is technically a lie, but we don't borrow anything directly from AstContext
// it is always from one of the contained bump allocators, which *are* stable
// however not deref. So we pretend to implement Deref, but we still guarantee
// stability of the references. This means that the yoke will not be invalidated
unsafe impl stable_deref_trait::StableDeref for AstContextHolder {}

pub struct File {
    yoke: yoke::Yoke<FileRef<'static>, AstContextHolder>,
}

impl File {
    pub fn as_ref(&self) -> FileRef<'_> {
        *self.yoke.get()
    }
}

pub fn parse(text: &[u8]) -> File {
    let yoke = yoke::Yoke::attach_to_cart(
        AstContextHolder {
            ctx: parser::AstContext::default(),
        },
        |ctx| parse_with(ctx, text),
    );

    File { yoke }
}

pub fn parse_with<'ast>(ctx: &'ast parser::AstContext, text: &[u8]) -> FileRef<'ast> {
    let items = parser::Parser::new(
        ctx,
        parser::ObjectPools::default(),
        &mut crate::parser::PanicDebugParseError,
        text,
    )
    .parse_file();

    FileRef { items }
}
