//! Buffer + shape primitive. Every op consumes and produces flat row-major `f32`.

/// A 2-D row-major shape `[rows, cols]`. The whole v0.50 op set is 2-D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Number of rows (e.g. `M` for activations, `N` for weights).
    pub rows: usize,
    /// Number of columns (e.g. `K`).
    pub cols: usize,
}

impl Shape {
    /// Construct a `[rows, cols]` shape.
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Shape { rows, cols }
    }
    /// Total element count `rows * cols`.
    #[must_use]
    pub const fn len(self) -> usize {
        self.rows * self.cols
    }
    /// `true` if the shape holds no elements.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.rows == 0 || self.cols == 0
    }
}
