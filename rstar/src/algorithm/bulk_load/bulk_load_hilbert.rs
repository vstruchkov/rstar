use crate::envelope::Envelope;
use crate::node::{ParentNode, RTreeNode};
use crate::object::RTreeObject;
use crate::params::RTreeParams;
use crate::point::{Point, RTreeNum};

#[cfg(not(test))]
use alloc::{vec, vec::Vec};

/// The largest number of dimensions the Hilbert keys are computed for.
const MAX_DIMENSIONS: usize = 8;

/// Packs elements bottom-up in the order of their centers along a Hilbert curve (experimental).
///
/// The scalars are only required to be ordered, so the curve runs through *rank space*: the
/// coordinate of a center on each axis is its rank among all centers on that axis, scaled to the
/// bits available per axis. Skewed data is thereby spread evenly over the curve.
///
/// Consecutive elements are grouped into nodes of `MAX_SIZE` children, distributed evenly so that
/// no node has fewer than about `MAX_SIZE / 2`, and the nodes of each level likewise into the next
/// one, until at most `MAX_SIZE` remain for the root. All leaves end up on the same level.
pub fn bulk_load_hilbert<T, Params>(elements: Vec<T>) -> ParentNode<T>
where
    T: RTreeObject,
    <T::Envelope as Envelope>::Point: Point,
    Params: RTreeParams,
{
    let dimensions = <T::Envelope as Envelope>::Point::DIMENSIONS;
    assert!(dimensions <= MAX_DIMENSIONS, "too many dimensions");
    let len = elements.len();

    let centers: Vec<_> = elements.iter().map(|e| e.envelope().center()).collect();
    let bits = if dimensions == 1 {
        63
    } else {
        (64 / dimensions as u32).min(32)
    };

    // Rank of every center on every axis, scaled to `bits`
    let mut coordinates = vec![0u64; len * dimensions];
    let mut order: Vec<usize> = (0..len).collect();
    for axis in 0..dimensions {
        order.sort_unstable_by_key(|&i| centers[i].nth(axis).ord());
        for (rank, &i) in order.iter().enumerate() {
            coordinates[i * dimensions + axis] = (((rank as u128) << bits) / len as u128) as u64;
        }
    }
    drop(centers);

    let mut keyed: Vec<(u64, T)> = elements
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            let mut x = [0u64; MAX_DIMENSIONS];
            x[..dimensions].copy_from_slice(&coordinates[i * dimensions..(i + 1) * dimensions]);
            (hilbert_key(&mut x[..dimensions], bits), e)
        })
        .collect();
    drop(coordinates);
    keyed.sort_unstable_by_key(|(key, _)| *key);

    let mut nodes: Vec<RTreeNode<T>> = keyed.into_iter().map(|(_, e)| RTreeNode::Leaf(e)).collect();
    while nodes.len() > Params::MAX_SIZE {
        nodes = pack::<T, Params>(nodes);
    }
    ParentNode::new_parent(nodes)
}

/// Groups consecutive nodes into parents of at most `MAX_SIZE` children, as evenly as possible.
fn pack<T, Params>(nodes: Vec<RTreeNode<T>>) -> Vec<RTreeNode<T>>
where
    T: RTreeObject,
    Params: RTreeParams,
{
    let len = nodes.len();
    let parents = len.div_ceil(Params::MAX_SIZE);
    let mut result = Vec::with_capacity(parents);
    let mut nodes = nodes.into_iter();
    for p in 0..parents {
        // Parent `p` takes the nodes from `p * len / parents` up to `(p + 1) * len / parents`
        let size = (p + 1) * len / parents - p * len / parents;
        let children: Vec<_> = nodes.by_ref().take(size).collect();
        result.push(RTreeNode::Parent(ParentNode::new_parent(children)));
    }
    result
}

/// The index along a Hilbert curve of a point given by `bits` bits per axis (John Skilling,
/// "Programming the Hilbert curve", 2004), with `x.len() * bits <= 64`.
fn hilbert_key(x: &mut [u64], bits: u32) -> u64 {
    let n = x.len();
    let m = 1u64 << (bits - 1);
    // Inverse undo excess work
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..n {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    // Gray encode
    for i in 1..n {
        x[i] ^= x[i - 1];
    }
    let mut t = 0;
    let mut q = m;
    while q > 1 {
        if x[n - 1] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for v in x.iter_mut() {
        *v ^= t;
    }
    // Interleave the transposed bits, most significant first
    let mut key = 0u64;
    for bit in (0..bits).rev() {
        for v in x.iter() {
            key = key << 1 | (v >> bit & 1);
        }
    }
    key
}

#[cfg(test)]
mod test {
    use super::hilbert_key;
    use crate::test_utilities::*;
    use crate::{DefaultParams, RTree};

    #[test]
    fn test_hilbert_key_is_a_curve() {
        // In 2D with 3 bits, consecutive keys are adjacent cells and every cell is visited once
        let mut cells = vec![(0u64, 0u64); 64];
        for x in 0..8 {
            for y in 0..8 {
                let key = hilbert_key(&mut [x, y], 3);
                cells[key as usize] = (x, y);
            }
        }
        for w in cells.windows(2) {
            let d = w[0].0.abs_diff(w[1].0) + w[0].1.abs_diff(w[1].1);
            assert_eq!(d, 1, "{w:?}");
        }
    }

    #[test]
    fn test_bulk_load_hilbert() {
        for size in (0..300).chain([1000, 4321]) {
            let points: Vec<[f64; 3]> = create_random_integers::<[i32; 3]>(size, SEED_1)
                .into_iter()
                .map(|p| p.map(f64::from))
                .collect();
            let mut tree = RTree::<_, DefaultParams>::bulk_load_hilbert_with_params(points.clone());
            assert_eq!(tree.size(), size);
            // Unlike OMT, packing keeps every node but the root within MIN_SIZE..=MAX_SIZE
            tree.root().sanity_check::<DefaultParams>(true);
            for p in &points {
                assert!(tree.contains(p));
            }
            for p in points.iter().take(size / 2) {
                assert!(tree.remove(p).is_some());
            }
            for p in create_random_integers::<[i32; 3]>(size, SEED_2) {
                tree.insert(p.map(f64::from));
            }
            tree.root().sanity_check::<DefaultParams>(false);
            assert_eq!(tree.size(), size - size / 2 + size);
        }
    }
}
