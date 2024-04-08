use std::mem::ManuallyDrop;

use smallvec::SmallVec;

pub(super) struct Pool<T, const N: usize> {
    items: Vec<SmallVec<[T; N]>>,
}

impl<T, const N: usize> Default for Pool<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Pool<T, N> {
    pub const fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn reuse<U, const M: usize>(self) -> Pool<U, M> {
        if core::alloc::Layout::new::<T>() != core::alloc::Layout::new::<U>() {
            return Pool::new();
        }

        let items = self
            .items
            .into_iter()
            .map(|item| {
                let vec = match item.into_inner() {
                    Ok(_) => {
                        debug_assert!(false);
                        unsafe { core::hint::unreachable_unchecked() }
                    }
                    Err(vec) => vec,
                };
                debug_assert_eq!(vec.len(), 0);

                let mut vec = ManuallyDrop::new(vec);
                let cap = vec.capacity();
                let len = vec.len();
                let ptr = vec.as_mut_ptr();

                if cap <= N {
                    debug_assert!(false);
                    unsafe { core::hint::unreachable_unchecked() }
                }
                SmallVec::from_vec(unsafe { Vec::from_raw_parts(ptr.cast(), len, cap) })
            })
            .filter(|x| x.capacity() > x.inline_size())
            .collect();

        Pool { items }
    }

    pub fn alloc(&mut self) -> SmallVec<[T; N]> {
        self.items.pop().unwrap_or_default()
    }

    pub fn free(&mut self, mut buffer: SmallVec<[T; N]>) {
        buffer.clear();

        if buffer.capacity() > buffer.inline_size() {
            self.items.push(buffer)
        }
    }
}

#[test]
fn test_reuse() {
    let mut pool = Pool::<u32, 2>::new();

    pool.free(SmallVec::with_capacity(10));

    let mut pool = pool.reuse::<f32, 2>();

    let items = pool.alloc();

    assert_eq!(items.capacity(), 10);
}
