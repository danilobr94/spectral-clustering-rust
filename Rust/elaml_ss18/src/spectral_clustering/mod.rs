// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

use ndarray::{ArrayBase, Array, Data, Ix2};
use num::{Num, NumCast, Zero, Signed, Float};
use rusty_machine::learning::{k_means::KMeansClassifier, UnSupModel};
use rusty_machine::prelude::BaseMatrix;
use rusty_machine;
use la::{Matrix, EigenDecomposition};
use adjacency_graph::knn::build;
use strings;


/// transforms weight matrix w into laplace matrix L = D - W
///
/// w: 2D Array weight matrix of the graph
fn into_laplace_matrix<A>(w: &mut Array<A,Ix2>)
	where
		A: Num + Copy + PartialOrd + Signed,
{
	assert_eq!(w.cols(), w.rows(), "{}",strings::SC_MATSQ_ERR );

	for i in 0..w.rows() {
		let mut d: A = Zero::zero();

	    for j in 0..w.cols() {
			d = d + w[[i,j]];
	    	w[[i,j]] = -w[[i,j]];
		}
		w[[i,i]] = w[[i,i]]+d;
	}
}

/// transforms weight matrix w into normalized laplace matrix: L = D^{-1/2} * (D-W) * D^{-1/2}
///
/// w: 2D Array weight matrix of the graph
fn into_normalized_laplace_matrix<A>(w: &mut Array<A,Ix2>)
    where
        A: Float + NumCast + Copy + PartialOrd + Signed,
{
	assert_eq!(w.cols(), w.rows(), "{}",strings::SC_MATSQ_ERR);

	let mut deg = Vec::new(); // the square root of the degree matrix (as vec to save memory)

	// get vector with node degrees ("diagonal degree matrix")
    for i in 0..w.rows() {
		let mut d: A = Zero::zero();

        for j in 0..w.cols() {
			d = d + w[[i,j]] ;
        }

        deg.push(d);
    }

	// compute L = D^{-1/2} *(D-W) * D^{-1/2}
	for i in 0..w.rows() {
		for j in 0..w.cols() {

			if i == j {
				w[[i, j]] = - (w[[i, j]] / deg[j]) + NumCast::from(1).unwrap();
			} else if w[[i, j]] > Zero::zero() {
				w[[i, j]] = - (w[[i, j]] / (deg[i]*deg[j]).sqrt());
			}
		}
	}
}

/// returns a vector of ascending numerals from (inclusive) from to (exclusive) to
///
/// from: first number in the vector(inclusive)
/// to: number following the last element
fn get_range(from:usize, to:usize) -> Vec<usize>{
	let mut ret = vec![0; to-from];
	let mut n = from;
	for i in 0..to-from{
		ret[i] = n;
		n+=1;
	}
	ret
}

/// Finds a good amount of clusters for a dataset by looking at the set's sorted eigenvalues
/// and finding the largest jump between two eigenvalues. Because the values are sorted, the
/// amount of eigenvalues after the largest jump is a good indicator of cluster number.
///
/// eig: A pointer to a vector of eigenvalues.
fn find_good_clusters(eigs: &Vec<f64>) -> usize{
	let mut cmp : Vec<(f64,usize)> = Vec::new();
	for i in 0..(eigs.len()-1){
		cmp.push((eigs[i] - eigs[i+1], i));
	}
	cmp.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let (_a,b) = cmp[0];
	eigs.len()-b+1
}


/// performs spectral clustering on graph
///
/// w: 2D Array weight matrix of the graph => "the graph". It unfortunately needs to be f64.
/// k: number of clusters in the data set
/// norm: Enum, whether normalized or un normalized spectral clustering is performed
///
/// returns vector of clusters [and changes w into laplace matrix!], and the eigenvalues
///
/// Spectral clustering (w as adjacency matrix):
/// 1. compute graph laplacian (normalized or unnormalized)
/// 2. decompose the laplacian into the k largest eigen vectors (=> spectral embedding)
/// 3. perform kmeans on the embedding (rows are the data points again)
///
pub fn cluster_from_graph(mut w: Array<f64,Ix2>, n_clusters: isize, normalized: bool) -> (Vec<usize>, Vec<f64>)
	//where
	//	A:  Copy + Signed + Float + ApproxEq<A>,
{
	assert_eq!(w.cols(), w.rows(), "{}" ,strings::SC_MATSQ_ERR);

	if normalized {
		into_normalized_laplace_matrix(&mut w); // transform w into laplace matrix
	} else {
		into_laplace_matrix(&mut w);
	}

	// eigen decomposition using la
	let mat = Matrix::new(w.rows(), w.cols(), w.into_raw_vec());
	let dec = EigenDecomposition::new(&mat);
	let v = dec.get_v(); // the eigen vectors => eigs are already sorted
	let eig = dec.get_real_eigenvalues();
	let eig=eig.to_vec();


	// ### alternative decomposition using rusty_machine::rulinalg
	//doesn't seem to work after a certain matrix size
	//let mat = rusty_machine::prelude::Matrix::new(w.rows(), w.cols(), w.into_raw_vec()); //for rulinalg
	//let (eig, v) = mat.eigendecomp().unwrap(); //for rulinalg //vectors are columns, data is row-major

	//n_clusters handling; if negative, attempt to find good value (cluster number probably low)
	let clusters : usize;
	if n_clusters < 0 {
		clusters = find_good_clusters(&eig);
	}else{
		clusters = n_clusters as usize;
	}

	// convert to rusty_machine matrix for kmeans
	let eigv = rusty_machine::prelude::Matrix::new(v.rows(), v.cols(), v.get_data().to_vec() );

	//get required k last eigenvector indexes
	let rng = get_range(v.cols()-clusters,v.cols());

	//KMeans using rusty_machine
	let mut model = KMeansClassifier::new(clusters);
	//unfortunately, train in rusty_machine is only implemented for f64
	model.train(&eigv.select_cols(&rng)).unwrap();
	let a = model.predict(&eigv.select_cols(&rng)).unwrap();
	(a.into_vec(), eig)
}

/// clusters data x with spectral clustering, returning labels and eigenvalues of the
/// spectral clustering Laplacian.
///
/// x: 2D array containing data points as rows
/// k_knn: number of neighbours for knn graph building. If negative, ~log(n) is used.
/// k_kmeans: number of clusters for clustering. If negative, algorithm attempts to find a good number.
/// normalized: If true, Laplacian is normalized
///
/// Steps:
/// 1. transform data into graph representation (here knn graph)
/// 2. Perform spectral clustering on this graph
///
pub fn cluster<A, S>(x: &ArrayBase<S,Ix2>, mut k_knn: isize, n_clusters: isize, normalized: bool) -> (Vec<usize>, Vec<f64>)
	where
		A: Num + NumCast + Copy,
		S: Data<Elem=A>,
{
	//if k_knn is negative, set it to integer log2 of the number of data points
	if k_knn < 0 {
		let  log = (x.rows() as f64).log2();
		k_knn = log as isize;
	}

	let knn = build(x, k_knn as usize);
	let (y, eigs ) = cluster_from_graph(knn, n_clusters, normalized);
	(y, eigs)
}


#[cfg(test)]
mod tests {
	use ndarray::arr2;
	use super::*;

	#[test]
	fn into_laplace_matrix_test() {
		let mut w =  arr2(&[
			[0, 1, 1, 2],
			[1, 0, 0, 3],
			[1, 0, 0, 0],
			[2, 3, 0, 0]]);

		let l =  arr2(&[
			[ 4, -1, -1, -2],
			[-1,  4,  0, -3],
			[-1,  0,  1,  0],
			[-2, -3,  0,  5]]);

		into_laplace_matrix(&mut w);

		assert_eq!(w, l);
	}

	#[test]
	fn into_normalized_laplace_matrix_test() {
		let mut w =  arr2(&[
			[0.0, 1.0, 0.0, 0.0, 0.0],
			[1.0, 0.0, 1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 1.0],
			[0.0, 0.0, 0.0, 1.0, 0.0]]);

		let x = - 1.0/2.0f64.sqrt();
		let y = -1.0/2.0f64;

		let l =  arr2(&[
			[1.0, x  , 0.0, 0.0, 0.0],
			[x  , 1.0, y  , 0.0, 0.0],
			[0.0, y  , 1.0, y  , 0.0],
			[0.0, 0.0, y  , 1.0, x  ],
			[0.0, 0.0, 0.0, x  , 1.0]]);

		into_normalized_laplace_matrix(&mut w);

		assert_eq!(w, l);
	}
}

