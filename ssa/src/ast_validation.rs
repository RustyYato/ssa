use ref_cast::RefCast;

use crate::ast::{self, Visit};

pub trait ValidationErrorReporter<'ast> {
    fn set_error(&mut self) {}

    fn report_invalid_expr_struct(&mut self, _ast: &'ast ast::ExprStruct<'ast>) {}
    fn report_invalid_expr_union(&mut self, _ast: &'ast ast::ExprUnion<'ast>) {}
    fn report_invalid_expr_enum(&mut self, _ast: &'ast ast::ExprEnum<'ast>) {}
}

#[repr(transparent)]
#[derive(ref_cast::RefCast)]
struct FileValidator<'ast, 'a> {
    errors: dyn ValidationErrorReporter<'ast> + 'a,
}

#[repr(transparent)]
#[derive(ref_cast::RefCast)]
struct NoTypesInSubExprSkipRoot<'ast, 'a> {
    errors: dyn ValidationErrorReporter<'ast> + 'a,
}

#[repr(transparent)]
#[derive(ref_cast::RefCast)]
struct NoTypesInSubExpr<'ast, 'a> {
    errors: dyn ValidationErrorReporter<'ast> + 'a,
}

pub fn validate<'ast>(
    file: &'ast [ast::Item<'ast>],
    errors: &mut dyn ValidationErrorReporter<'ast>,
) {
    file.visit(FileValidator::ref_cast_mut(errors))
}

impl<'ast> ast::Visitor<'ast> for FileValidator<'ast, '_> {
    fn visit_item_let(&mut self, _id: ast::ItemId, stmt_let: &'ast ast::Let<'ast>) {
        if let Some(value) = &stmt_let.value {
            value.default_visit(NoTypesInSubExprSkipRoot::ref_cast_mut(&mut self.errors));
        }
        stmt_let.default_visit(self);
    }
}

impl<'ast> ast::Visitor<'ast> for NoTypesInSubExprSkipRoot<'ast, '_> {
    fn visit_expr(&mut self, expr: &'ast ast::Expr<'ast>) {
        expr.default_visit(NoTypesInSubExpr::ref_cast_mut(&mut self.errors));
    }
}

impl<'ast> ast::Visitor<'ast> for NoTypesInSubExpr<'ast, '_> {
    fn visit_expr_struct(&mut self, expr_struct: &'ast ast::ExprStruct<'ast>) {
        self.errors.set_error();
        self.errors.report_invalid_expr_struct(expr_struct);
    }

    fn visit_expr_union(&mut self, expr_struct: &'ast ast::ExprUnion<'ast>) {
        self.errors.set_error();
        self.errors.report_invalid_expr_union(expr_struct);
    }

    fn visit_expr_enum(&mut self, expr_struct: &'ast ast::ExprEnum<'ast>) {
        self.errors.set_error();
        self.errors.report_invalid_expr_enum(expr_struct);
    }
}

#[test]
fn test() {
    #[derive(Default)]
    struct HasErrors(bool);

    impl ValidationErrorReporter<'_> for HasErrors {
        fn set_error(&mut self) {
            self.0 = true;
        }
    }
    let mut errors = HasErrors::default();
    validate(&[], &mut errors);
    assert!(!errors.0);

    validate(
        &[ast::Item {
            id: ast::ItemId::from_u32(1),
            kind: ast::ItemKind::Let(&ast::Let {
                binding: ast::Ident {
                    id: ast::IdentId::from_u32(1),
                    name: istr::IStr::new("Option"),
                },
                ty: None,
                value: Some(ast::Expr {
                    id: ast::ExprId::from_u32(1),
                    kind: ast::ExprKind::Union(&ast::ExprUnion {
                        params: &[ast::TypeParam {
                            name: ast::Ident {
                                id: ast::IdentId::from_u32(2),
                                name: istr::IStr::new("T"),
                            },
                            bounds: [],
                        }],
                        variants: &[
                            ast::Field {
                                name: ast::Ident {
                                    id: ast::IdentId::from_u32(2),
                                    name: istr::IStr::new("Some"),
                                },
                                ty: ast::Type {
                                    id: ast::TypeId::from_u32(1),
                                    kind: ast::TypeKind::Concrete(&ast::TypeConcrete {
                                        name: ast::Path {
                                            segments: &[ast::Ident {
                                                id: ast::IdentId::from_u32(3),
                                                name: istr::IStr::new("T"),
                                            }],
                                        },
                                        generics: &[],
                                    }),
                                },
                            },
                            ast::Field {
                                name: ast::Ident {
                                    id: ast::IdentId::from_u32(4),
                                    name: istr::IStr::new("None"),
                                },
                                ty: ast::Type {
                                    id: ast::TypeId::from_u32(2),
                                    kind: ast::TypeKind::Primitive(ast::TypePrimitive::Unit),
                                },
                            },
                        ],
                    }),
                }),
            }),
        }],
        &mut errors,
    );
    assert!(!errors.0);
}
