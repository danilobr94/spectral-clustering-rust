// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

use adjacency_graph::p_norm::p_norm_dist;
use ndarray::{Axis, Array2, ArrayBase, Data, Ix2};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use num::{Num, NumCast};

/// struct for (index, distance) pairs
struct IdxDist {
    ind: usize,	// index of point in data matrix
    dist: f64,	// distance to reference point
}

/// implement Ord trait for IdxDist to work with IdxDist
///
/// only the distance dist is compared
impl Ord for IdxDist {
    fn cmp(&self, other: &IdxDist) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

impl Eq for IdxDist {}

impl PartialOrd for IdxDist {
    fn partial_cmp(&self, other: &IdxDist) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for IdxDist {
    fn eq(&self, other: &IdxDist) -> bool {
        self.dist == other.dist
    }
}


/// returns binary heap of (point, distance) struct for k closest points relative to row i in x
/// the point is the key, and value the distance
///
/// x: 2D array containing data points as rows
/// i: index of point for which k closest distances are returned
/// k: number of closest distances to return
fn get_k_closest_idx_dst<A, S>(x: &ArrayBase<S,Ix2>, i: usize ,k: usize) -> BinaryHeap<IdxDist>
    where
        A: Num + NumCast + Copy,
        S: Data<Elem=A>,
{
    let mut k_closest: BinaryHeap<IdxDist> = BinaryHeap::new();
    k_closest.reserve_exact(k);	// reserve memory for heap, to avoid reallocation

    // compute distance between point i and all other points and store the k closest points in a tree
    for j in 0..x.rows(){
        if i != j {
            let d = p_norm_dist (&x.subview(Axis(0), i), &x.subview(Axis(0), j), 2.0);

            // if less then k elements in tree, add the new one
            if k_closest.len() < k {
                k_closest.push(IdxDist { ind: j, dist: d });

                // else only add closer elements
            } else if d < k_closest.peek().unwrap().dist {
                k_closest.push(IdxDist { ind: j, dist: d });    // insert new item
                k_closest.pop();                            // remove previously largest item
            }
        }
    }
    k_closest
}


/// builds knn adjacency matrix for input data x
///
/// x: 2D array containing data points as rows
/// k: number of neighbours for knn graph
pub fn build<A, S>(x: &ArrayBase<S,Ix2>, k: usize) -> Array2<f64>
    where
        A: Num + NumCast + Copy,
        S: Data<Elem=A>,
{

    let mut w = Array2::<f64>::zeros((x.rows(),x.rows())); // adjacency matrix initialized with zeros

    // for each point in x, compute k closest neighbours
    for i in 0..x.rows() {
        let kn = get_k_closest_idx_dst(&x, i, k);	// returns array of IdxDist struct (index and distance of k closest points)

        for j in kn.iter() {		// set adjacency of k closest points to distance
            w[[i, j.ind]] = j.dist;
            w[[j.ind, i]] = j.dist;
        }
    }
    w
}

///builds a knn graph with both a maximum number of neighbors and a distance cut-off epsilon
///
/// x: 2D array containing data points as rows
/// k: number of neighbours for knn graph
/// epsilon: distance cut-off.
pub fn build_knn_with_epsilon<A, S>(x: &ArrayBase<S,Ix2>, k: usize, epsilon: f64) -> Array2<f64>
    where
        A: Num + NumCast + Copy,
        S: Data<Elem=A>,
{
    let mut w = Array2::<f64>::zeros((x.rows(),x.rows())); // adjacency matrix initialized with zeros

    // for each point in x, compute k closest neighbours
    for i in 0..x.rows() {
        let kn = get_k_closest_idx_dst(&x, i, k);	// returns array of IdxDist struct (index and distance of k closest points)

        for j in kn.iter() {		// set adjacency of k closest points to distance
            if j.dist <= epsilon {
                w[[i, j.ind]] = j.dist;
                w[[j.ind, i]] = j.dist;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // helper function to check if heap contains correct points
    fn heap_eq(mut b: BinaryHeap<IdxDist>, l: &[usize]) -> bool{
        for _i in 0..b.len(){
            if l.contains(&b.peek().unwrap().ind) {
                b.pop();
            }
        }
        b.is_empty()
    }

    #[test]
    fn get_k_closest_idx_dst_test() {
        let a = arr2(&[[1,2],[1,1],[0,2],[2,2],[3,5],[1,-2]]);
        let b = get_k_closest_idx_dst(&a,1, 4);
        let res = [0usize,2,3, 5];
        assert!(heap_eq(b,&res));
    }

    #[test]
    fn build_knn_adj_test() {
        let a = arr2(&[[1,2],[0,2],[2,2],[2,-2]]);
        let b = build(&a,2);
        let res =  arr2(&[
            [0.0, 1.0, 1.0, 17f64.sqrt()],
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0, 4.0],
            [17f64.sqrt(), 0.0, 4.0, 0.0]]);

        assert_eq!(b, res);
    }

    #[test]
    fn build_knn_epsilon_adj_test() {
        let a = arr2(&[[1,2],[0,2],[2,2],[2,-2]]);
        let b =build_knn_with_epsilon(&a, 3, 4.);
        let res =  arr2(&[
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0, 4.0],
            [0.0, 0.0, 4.0, 0.0]]);

        assert_eq!(b, res);
    }
}