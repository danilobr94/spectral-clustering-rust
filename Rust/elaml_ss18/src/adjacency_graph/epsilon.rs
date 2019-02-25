// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

use adjacency_graph::p_norm::p_norm_dist;
use ndarray::{Axis, Array2, ArrayBase, Data, Ix2};
use num::{Num, NumCast};


/// struct for (index, distance) pairs
struct IdxDist {
    ind: usize,	// index of point in data matrix
    dist: f64,	// distance to reference point
}

/// returns vec<IdxDist> of edges with distances smaller epsilon
/// the point is the key, and value the distance
///
/// x: 2D array containing data points as rows
/// i: index of point for which k closest distances are returned
/// epsilon: threshold distance
fn get_edges<A, S>(x: &ArrayBase<S,Ix2>, i: usize, epsilon: f64) -> Vec<IdxDist>
    where
        A: Num + NumCast + Copy,
        S: Data<Elem=A>,
{
    let mut v = Vec::new();

    // compute distance between point i and all other points and store the k closest points in a tree
    for j in 0..x.rows(){
        if i != j {
            let d = p_norm_dist(&x.subview(Axis(0), i), &x.subview(Axis(0), j), 2.0);

            if d < epsilon{
                v.push(IdxDist { ind: j, dist: d });
            }
        }
    }
    v
}


/// builds epsilon adjacency matrix for input data x
///
/// x: 2D array containing data points as rows
/// epsilon: threshold distance
pub fn build<A, S>(x: &ArrayBase<S,Ix2>, epsilon: f64) -> Array2<f64>
    where
        A: Num + NumCast + Copy,
        S: Data<Elem=A>,
{
    let mut w = Array2::<f64>::zeros((x.rows(),x.rows())); // adjacency matrix initialized with zeros

    // for each point in x, compute k closest neighbours
    for i in 0..x.rows() {
        let kn = get_edges(&x, i, epsilon);	// returns vec of IdxDist struct (index and distance of k closest points)

        for j in kn.iter() {		// set adjacency of k closest points to distance
            w[[i, j.ind]] = j.dist;
            w[[j.ind, i]] = j.dist;
        }
    }
    w
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2};

    #[test]
    fn build_knn_adj_test() {
        let a = arr2(&[[1,2],[0,2],[2,2],[2,-2]]);
        let b = build(&a,1.5);
        let res =  arr2(&[
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]);

        assert_eq!(b, res);
    }
}