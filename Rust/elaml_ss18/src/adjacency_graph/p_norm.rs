// Course:      Efficient Linear Algebra and Machine Learning
// Project: 	Spectral Clustering
// Author:      Danilo Brajovic and Tristan Baumann
//
// Honor Code:  I pledge that this program represents my own work.

use ndarray::{ArrayBase, Data, Ix1};
use num::{Num, NumCast};
use strings;


/// p-norm on distance between points x and y
/// x,y: 1D ndarrays of same length
/// p: p for norm (>= 1.0)
pub fn p_norm_dist<A, S>(x: &ArrayBase<S,Ix1>, y: &ArrayBase<S,Ix1>, p: f64) -> f64
    where
        A: Num + NumCast  + Copy,
        S: Data<Elem=A>,
{
    assert_eq!(x.len(),y.len(), "{}", strings::PNORM_DIM_ERR);
    assert!(p>=1.0, "{}", strings::PNORM_NDEF_ERR);

    let mut s : f64 = 0.0;

    for i in 0..x.len() {
        let d:f64 = NumCast::from(x[i]-y[i]).unwrap();
        s += d.abs().powf(p);
    }
    s.powf(1.0/p) // the p-th root of s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn p1_norm_test() {
        let a = arr1(&[0.,0.,1.]);
        let b = arr1(&[2.,1.,7.]);
        let c = p_norm_dist(&a,&b, 1.0);
        assert_eq!(c, 9.0f64);
    }

    #[test]
    fn p2_norm_test() {
        let a = arr1(&[1,2,3]);
        let b = arr1(&[5,5,8]);
        let c = p_norm_dist(&a,&b, 2.0);
        assert_eq!(c, 50.0_f64.sqrt());
    }

    #[test]
    fn p3_norm_test() {
        let a = arr1(&[0f32,0f32,0f32]);
        let b = arr1(&[1f32,2f32,3f32]);
        let c = p_norm_dist(&a,&b, 3.0);
        assert_eq!(c, 36.0_f64.powf(1.0/3.0));
    }
}