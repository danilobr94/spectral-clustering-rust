// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

extern crate elaml_ss18;
extern crate ndarray;
extern crate plotlib;

extern crate rusty_machine;

use plotlib::scatter::Scatter;
use plotlib::scatter;
use plotlib::style::{Marker, Point};
use plotlib::view::View;
use plotlib::page::Page;

use elaml_ss18::spectral_clustering;
use elaml_ss18::load_data;

use rusty_machine::learning::{k_means::KMeansClassifier, UnSupModel};

/// creates scatter of x with data labeled according to y
fn plt_scatter(x: &ndarray::Array2<f64>, y: Vec<usize>, p: String){
    assert_eq!(x.rows(), y.len());

    let mut data0 = Vec::new();
    let mut data1 = Vec::new();
    let mut data2 = Vec::new();


    for i in  0..x.rows() {
        if y[i] == 0{
            data0.push((x[[i, 0]], x[[i, 1]]));
        } else if y[i] == 1 {
            data1.push((x[[i, 0]], x[[i, 1]]));
        } else {
            data2.push((x[[i, 0]], x[[i, 1]]));
        }
    }

    let s0 = Scatter::from_vec(&data0)
        .style(scatter::Style::new()
            .marker(Marker::Square) // setting the marker to be a square
            .colour("#DD3355")); // and a custom colour


    let s1 = Scatter::from_vec(&data1)
        .style(scatter::Style::new() // uses the default marker
            .colour("#35C788")); // and a different colour

    let s2 = Scatter::from_vec(&data2)
        .style(scatter::Style::new() // uses the default marker
            .colour("#000000")); // and a different colour


    // The 'view' describes what set of data is drawn
    let v = View::new()
        .add(&s0)
        .add(&s1)
        .add(&s2)
        .x_range(-1., 1.)
        .y_range(-1., 1.);

    // A page with a single view is then saved to an SVG file
    Page::single(&v).save(p);
}

/// main
fn main() {

    //Spectral Clustering
	let x = load_data::load_f64("3rings-simple.txt".to_string(), " ".to_string(),2);
    let (y, _eigs) = spectral_clustering::cluster(&x, 8, -1, true);
    println!("Spectral Clustering Result: Labels: {:?}", y);
    println!();

    plt_scatter(&x, y, "3rings-simple.svg".to_string());


    //And k-means for comparison
    let x: ndarray::Array<f64,ndarray::Ix2> = load_data::load_f64("3rings-simple.txt".to_string(), " ".to_string(),2);
    let dataset = rusty_machine::prelude::Matrix::new(x.rows(), x.cols(),  x.into_raw_vec());
    let mut model = KMeansClassifier::new(3);
    model.train(&dataset).unwrap();
    let a = model.predict(&dataset).unwrap();
    let x: ndarray::Array<f64,ndarray::Ix2> = load_data::load_f64("3rings-simple.txt".to_string(), " ".to_string(),2);

    plt_scatter(&x, a.into_vec(), "3rings-kmeans.svg".to_string());

}
