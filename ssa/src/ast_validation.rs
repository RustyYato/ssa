use crate::ast;

mod name_resolution;
mod validate_type_def;

pub use name_resolution::resolve_names;

pub trait ValidationErrorReporter<'ast> {
    fn set_error(&mut self) {}

    fn report_invalid_expr_struct(&mut self, _ast: &'ast ast::ExprStruct<'ast>) {}
    fn report_invalid_expr_union(&mut self, _ast: &'ast ast::ExprUnion<'ast>) {}
    fn report_invalid_expr_enum(&mut self, _ast: &'ast ast::ExprEnum<'ast>) {}
}

pub fn validate<'ast>(
    file: &'ast [ast::Item<'ast>],
    errors: &mut dyn ValidationErrorReporter<'ast>,
) {
    validate_type_def::validate_type_defs(file, errors);
}
