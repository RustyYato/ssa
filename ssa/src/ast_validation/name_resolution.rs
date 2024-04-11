use std::collections::HashMap;

use crate::ast::{self, Visit};

struct CollectGlobals {
    names: istr::IStrMap<ast::IdentId>,
}

struct NameResolver {
    globals: istr::IStrMap<ast::IdentId>,

    names_map: istr::IStrMap<Vec<ast::IdentId>>,
    names_stack: Vec<istr::IStr>,

    id_map: rustc_hash::FxHashMap<ast::IdentId, ast::IdentId>,
}

pub struct Resolution {
    id_map: rustc_hash::FxHashMap<ast::IdentId, ast::IdentId>,
}

impl Resolution {
    pub fn get(&self, id: ast::IdentId) -> Option<ast::IdentId> {
        self.id_map.get(&id).copied()
    }
}

pub fn resolve_names(items: &[ast::Item<'_>]) -> Resolution {
    let mut globals = CollectGlobals {
        names: HashMap::default(),
    };

    items.visit(&mut globals);

    let mut globals = NameResolver {
        globals: globals.names,

        names_map: istr::IStrMap::default(),
        names_stack: Vec::new(),

        id_map: HashMap::default(),
    };

    items.visit(&mut globals);

    Resolution {
        id_map: globals.id_map,
    }
}

impl<'ast> ast::Visitor<'ast> for CollectGlobals {
    fn visit_item_let(&mut self, _id: ast::ItemId, stmt_let: &'ast ast::Let<'ast>) {
        let old = self
            .names
            .insert(stmt_let.binding.name, stmt_let.binding.id);

        if old.is_some() {
            panic!("Duplicate global variable")
        }
    }
}

impl<'ast> ast::Visitor<'ast> for NameResolver {
    fn visit_item_let(&mut self, _id: ast::ItemId, stmt_let: &'ast ast::Let<'ast>) {
        stmt_let.value.visit(self);
    }

    fn visit_let(&mut self, item_let: &'ast ast::Let<'ast>) {
        item_let.value.visit(self);

        self.names_map
            .entry(item_let.binding.name)
            .or_default()
            .push(item_let.binding.id);
        self.names_stack.push(item_let.binding.name);
    }

    fn visit_expr_ident(&mut self, _id: ast::ExprId, expr_ident: &'ast ast::Ident) {
        match self
            .names_map
            .get(&expr_ident.name)
            .and_then(|stack| stack.last())
            .copied()
            .or_else(|| self.globals.get(&expr_ident.name).copied())
        {
            Some(source) => {
                self.id_map.insert(expr_ident.id, source);
            }
            None => {
                panic!("unresolved {}", expr_ident.name)
            }
        }
    }

    fn visit_block(&mut self, block: &'ast ast::Block<'ast>) {
        let current = self.names_stack.len();

        block.default_visit(self);

        for name in self.names_stack.drain(current..) {
            let id = self
                .names_map
                .get_mut(&name)
                .expect("All names must have been registered before being put in the stack")
                .pop();
            debug_assert!(id.is_some());
        }
    }
}

#[test]
fn test_basic_name_resolution() {
    let file = crate::parse(b"let x = y; let y = 0; let z = { let x = x; let y = x; };");

    let res = resolve_names(file.as_ref().items);

    assert_eq!(
        res.id_map,
        HashMap::from_iter([
            (ast::IdentId::from_u32(8), ast::IdentId::from_u32(5)),
            (ast::IdentId::from_u32(2), ast::IdentId::from_u32(3)),
            (ast::IdentId::from_u32(6), ast::IdentId::from_u32(1)),
        ])
    )
}
