use crate::envelope::Envelope;
use crate::node::{ParentNode, RTreeNode};
use crate::object::RTreeObject;
use crate::params::RTreeParams;
use crate::point::Point;

#[cfg(not(test))]
use alloc::vec::Vec;

/// Builds the subtree of the given `height` over the last `len` elements, removing them from
/// `elements`.
///
/// The height is fixed by the caller (not derived from `len` again), which keeps all leaves on
/// the same level.
fn bulk_load_recursive<T, Params>(elements: &mut Vec<T>, len: usize, height: usize) -> ParentNode<T>
where
    T: RTreeObject,
    <T::Envelope as Envelope>::Point: Point,
    Params: RTreeParams,
{
    let start = elements.len() - len;
    if height <= 1 {
        // Reached leaf level. Draining allocates the node with its exact size.
        let leaves = elements.drain(start..).map(RTreeNode::Leaf).collect();
        return ParentNode::new_parent(leaves);
    }
    // The number of elements each subtree can hold
    let subtree_capacity = Params::MAX_SIZE.saturating_pow(height as u32 - 1);
    // How many clusters will this node contain at least
    let clusters = len.div_ceil(subtree_capacity);
    // More clusters would leave some subtree less than half full
    let max_clusters = (len / subtree_capacity.div_ceil(2)).min(Params::MAX_SIZE);
    let mut children = Vec::with_capacity(max_clusters);
    partition_into_clusters::<_, Params>(
        elements,
        len,
        clusters,
        max_clusters,
        <T::Envelope as Envelope>::Point::DIMENSIONS,
        height - 1,
        &mut children,
    );
    // Clusters are consumed from the end of `elements`
    children.reverse();
    ParentNode::new_parent(children)
}

/// Partitions the last `len` elements into `clusters` clusters of (almost) equal size and
/// appends the subtree of each cluster to `children`, last cluster first.
///
/// The elements are split into slabs along one axis, every slab holding a whole number of
/// clusters, and the slabs are split further along the `dims_left - 1` remaining axes. If
/// `max_clusters` allows for it, the number of clusters is raised to fill the grid of slabs:
/// clusters of the same shape overlap less than a grid with a row of wider clusters.
fn partition_into_clusters<T, Params>(
    elements: &mut Vec<T>,
    len: usize,
    mut clusters: usize,
    max_clusters: usize,
    dims_left: usize,
    subtree_height: usize,
    children: &mut Vec<RTreeNode<T>>,
) where
    T: RTreeObject,
    <T::Envelope as Envelope>::Point: Point,
    Params: RTreeParams,
{
    if clusters == 1 {
        let subtree = bulk_load_recursive::<_, Params>(elements, len, subtree_height);
        children.push(RTreeNode::Parent(subtree));
        return;
    }
    // Try to split all clusters among the remaining dimensions as evenly as possible by taking
    // the nth root. On the last axis, every slab is a cluster.
    let slabs = ceil_root(clusters, dims_left);
    let grid = slabs * clusters.div_ceil(slabs);
    if grid <= max_clusters {
        clusters = grid;
    }
    let slab_clusters_end = |slab: usize| slab * clusters / slabs;
    let slab_end = |slab: usize| slab_clusters_end(slab) * len / clusters;

    // The first axis gets the most slabs. Rotating it with the level of the tree keeps the
    // envelopes of the nodes from being stretched along the same axis on every level.
    let axis = (dims_left - 1 + subtree_height) % <T::Envelope as Envelope>::Point::DIMENSIONS;
    let start = elements.len() - len;
    select_slabs(&mut elements[start..], axis, 0, slabs, &slab_end);
    for slab in (0..slabs).rev() {
        let slab_clusters = slab_clusters_end(slab + 1) - slab_clusters_end(slab);
        partition_into_clusters::<_, Params>(
            elements,
            slab_end(slab + 1) - slab_end(slab),
            slab_clusters,
            slab_clusters,
            dims_left - 1,
            subtree_height,
            children,
        );
    }
}

/// Reorders `elements`, which hold the slabs `first..last`, so that every slab only contains
/// elements that are not larger along `axis` than the elements of the following slabs.
fn select_slabs<T>(
    elements: &mut [T],
    axis: usize,
    first: usize,
    last: usize,
    slab_end: &impl Fn(usize) -> usize,
) where
    T: RTreeObject,
{
    if last - first < 2 {
        return;
    }
    let middle = first + (last - first) / 2;
    let partition_point = slab_end(middle) - slab_end(first);
    T::Envelope::partition_envelopes(axis, elements, partition_point);
    let (lower, upper) = elements.split_at_mut(partition_point);
    select_slabs(lower, axis, first, middle, slab_end);
    select_slabs(upper, axis, middle, last, slab_end);
}

/// Returns the smallest `root` with `root.pow(degree) >= value`.
fn ceil_root(value: usize, degree: usize) -> usize {
    let mut root = 1usize;
    while root.saturating_pow(degree as u32) < value {
        root += 1;
    }
    root
}

/// A multi dimensional implementation of the OMT bulk loading algorithm.
///
/// See http://ceur-ws.org/Vol-74/files/FORUM_18.pdf
///
/// All leaves of the resulting tree are on the same level and every node but the root is at
/// least half full.
pub fn bulk_load_sequential<T, Params>(mut elements: Vec<T>) -> ParentNode<T>
where
    T: RTreeObject,
    <T::Envelope as Envelope>::Point: Point,
    Params: RTreeParams,
{
    // The height of the resulting tree, assuming all nodes will be filled up to MAX_SIZE
    let mut height = 1;
    let mut capacity = Params::MAX_SIZE;
    while capacity < elements.len() {
        capacity = capacity.saturating_mul(Params::MAX_SIZE);
        height += 1;
    }
    let len = elements.len();
    bulk_load_recursive::<_, Params>(&mut elements, len, height)
}

#[cfg(test)]
mod test {
    use crate::test_utilities::*;
    use crate::{Point, RTree, RTreeObject};
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::hash::Hash;

    #[test]
    fn test_bulk_load_small() {
        let random_points = create_random_integers::<[i32; 2]>(50, SEED_1);
        create_and_check_bulk_loading_with_points(&random_points);
    }

    #[test]
    fn test_bulk_load_large() {
        let random_points = create_random_integers::<[i32; 2]>(3000, SEED_1);
        create_and_check_bulk_loading_with_points(&random_points);
    }

    #[test]
    fn test_bulk_load_with_different_sizes() {
        for size in (0..100).map(|i| i * 7) {
            test_bulk_load_with_size_and_dimension::<[i32; 2]>(size);
            test_bulk_load_with_size_and_dimension::<[i32; 3]>(size);
            test_bulk_load_with_size_and_dimension::<[i32; 4]>(size);
        }
    }

    fn test_bulk_load_with_size_and_dimension<P>(size: usize)
    where
        P: Point<Scalar = i32> + RTreeObject + Send + Sync + Eq + Clone + Debug + Hash + 'static,
        P::Envelope: Send + Sync,
    {
        let random_points = create_random_integers::<P>(size, SEED_1);
        create_and_check_bulk_loading_with_points(&random_points);
    }

    fn create_and_check_bulk_loading_with_points<P>(points: &[P])
    where
        P: RTreeObject + Send + Sync + Eq + Clone + Debug + Hash + 'static,
        P::Envelope: Send + Sync,
    {
        let tree = RTree::bulk_load(points.into());
        let set1: HashSet<_> = tree.iter().collect();
        let set2: HashSet<_> = points.iter().collect();
        assert_eq!(set1, set2);
        assert_eq!(tree.size(), points.len());
    }

    /// All leaves must be on the same level and all nodes sufficiently filled, for any size.
    /// (OMT used to derive the depth of every subtree from its own size, which put leaves of
    /// e.g. 25 elements onto different levels and made later insertions panic.)
    #[test]
    fn test_bulk_load_is_balanced() {
        use crate::params::{DefaultParams, RTreeParams};
        use crate::RStarInsertionStrategy;

        struct SmallParams;
        impl RTreeParams for SmallParams {
            const MIN_SIZE: usize = 2;
            const MAX_SIZE: usize = 4;
            const REINSERTION_COUNT: usize = 1;
            type DefaultInsertionStrategy = RStarInsertionStrategy;
        }

        struct OddParams;
        impl RTreeParams for OddParams {
            const MIN_SIZE: usize = 4;
            const MAX_SIZE: usize = 7;
            const REINSERTION_COUNT: usize = 2;
            type DefaultInsertionStrategy = RStarInsertionStrategy;
        }

        fn check<P, Params>(size: usize)
        where
            P: Point<Scalar = i32> + RTreeObject,
            Params: RTreeParams,
        {
            let points = create_random_integers::<P>(size, SEED_1);
            let tree = RTree::<P, Params>::bulk_load_with_params(points);
            assert_eq!(tree.size(), size);
            tree.root().sanity_check::<Params>(true);
        }

        let sizes = (0..300).chain((300..1300).step_by(7));
        for size in sizes.chain([1296, 1297, 6145, 7167, 7776, 7777]) {
            check::<[i32; 2], DefaultParams>(size);
            check::<[i32; 2], SmallParams>(size);
            check::<[i32; 2], OddParams>(size);
            check::<[i32; 3], DefaultParams>(size);
            check::<[i32; 4], SmallParams>(size);
        }
    }

    #[test]
    fn test_insert_and_remove_after_bulk_load() {
        for size in [25, 27, 97, 111, 385, 447, 1537, 1791] {
            let rectangles = create_random_rectangles(size, SEED_1);
            let more_rectangles = create_random_rectangles(size, SEED_2);
            let mut tree = RTree::bulk_load(rectangles.clone());
            for (old, new) in rectangles.iter().zip(&more_rectangles) {
                assert!(tree.remove(old).is_some());
                tree.insert(*new);
            }
            assert_eq!(tree.size(), size);
            assert!(more_rectangles.iter().all(|r| tree.contains(r)));
        }
    }

    /// Verify that bulk-loaded tree nodes don't retain excessive Vec capacity.
    ///
    /// Without shrinking over-sized allocations during partitioning, Rust's
    /// in-place collect optimization (triggered when
    /// `size_of::<T>() == size_of::<RTreeNode<T>>()`) can preserve them in
    /// the final tree nodes. For large inputs this can waste many gigabytes
    /// of memory.
    #[test]
    fn test_bulk_load_no_excess_capacity() {
        use crate::node::RTreeNode;

        const N: usize = 10_000;
        let points: Vec<[i32; 2]> = (0..N as i32).map(|i| [i, i * 3]).collect();
        let tree = RTree::bulk_load(points);
        assert_eq!(tree.size(), N);

        // Walk all internal nodes and check that children Vecs are not
        // drastically over-allocated. Allow 2x as headroom for normal
        // allocator rounding.
        let max_allowed_ratio = 2.0_f64;
        let mut checked = 0usize;
        let mut stack: Vec<&crate::node::ParentNode<[i32; 2]>> = vec![tree.root()];
        while let Some(node) = stack.pop() {
            let len = node.children.len();
            let cap = node.children.capacity();
            assert!(len > 0, "empty internal node should not exist");
            let ratio = cap as f64 / len as f64;
            assert!(
                ratio <= max_allowed_ratio,
                "node children Vec has excessive capacity: len={len}, cap={cap}, ratio={ratio:.1}x \
                 (max {max_allowed_ratio}x). This indicates split_off over-capacity is leaking \
                 into the tree."
            );
            checked += 1;
            for child in &node.children {
                if let RTreeNode::Parent(ref p) = child {
                    stack.push(p);
                }
            }
        }
        assert!(checked > 1, "expected multiple internal nodes for N={N}");
    }
}
