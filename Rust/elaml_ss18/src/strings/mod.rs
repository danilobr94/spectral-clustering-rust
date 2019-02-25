// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

///This file contains error message strings
///

pub static PNORM_DIM_ERR : &str = "p_norm::p_norm_dist() failed: Input vector dimensions do not match.";
pub static PNORM_NDEF_ERR : &str = "p_norm::p_norm_dist() failed: p-norm not defined for p < 1.0 .";
pub static SC_MATSQ_ERR : &str = "Weight matrix needs to be square.";