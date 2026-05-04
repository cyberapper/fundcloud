//! Concrete pattern detectors.
//!
//! v1 ships the tier-1 reversal subset: head-and-shoulders pair, double
//! top/bottom, triple top/bottom, ascending/descending/symmetrical
//! triangle. Each family lives in its own submodule and exposes the
//! detector struct(s) needed.

pub mod double;
pub mod head_shoulders;
pub mod triangles;
pub mod triple;

pub use double::{DoubleBottomDetector, DoubleTopDetector};
pub use head_shoulders::{HeadShouldersDetector, InverseHeadShouldersDetector};
pub use triangles::{
    AscendingTriangleDetector, DescendingTriangleDetector, SymmetricalTriangleDetector,
};
pub use triple::{TripleBottomDetector, TripleTopDetector};
