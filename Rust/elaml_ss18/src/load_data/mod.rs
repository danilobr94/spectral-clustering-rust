// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

use ndarray::Array2;
use std::fs::File;
use stdinout::OrExit;
use std::io::{BufReader, BufRead};

/// reads data from file and creates an ndarray matrix
///
/// path: file path
/// delimiter: data delimiter
/// n_cols: number of columns the output matrix will have (should be the same as data dimensionality)
pub fn load_f64(path: String, delimiter: String, n_cols: usize) -> Array2<f64>{

	let file = File::open(path).or_exit("Cannot open file...", 1);
	let lines = BufReader::new(file).lines();
	let mut data_vec : Vec<f64> = Vec::new();

	for l in lines{
		let line = l.unwrap();
		let mut split = line.split(&delimiter);			

		for s in split{
			if ! s.parse::<f64>().is_err(){
			    data_vec.push(s.parse::<f64>().unwrap());
			}
		}
	}

	let shape_vec = (data_vec.len() / n_cols, n_cols);
	let x = Array2::<f64>::from_shape_vec(shape_vec, data_vec)
		.or_exit("Loading file into matrix failed.\nPossibly because data dimensionality does not match n_cols\nor data cannot be converted into f64.",1);
	x
}
