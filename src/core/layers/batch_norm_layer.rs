use ndarray::{Array2, Axis};
use ndarray_rand::{rand_distr::{Normal, Uniform}, RandomExt};

use crate::core::layer::Layer;



#[derive(Debug)]
pub struct BatchNormalizationLayer {
    training: bool,
    eps: f32,
    momentum: f32,
    gamma: Array2<f32>,
    beta: Array2<f32>,
    running_mean: Array2<f32>,
    running_var: Array2<f32>,
    grad: Option<Array2<f32>>,    
    output: Option<Array2<f32>>,
    input: Option<Array2<f32>>
}

impl BatchNormalizationLayer {
    pub fn new(dimensions: usize, eps: f32, momentum: f32) -> BatchNormalizationLayer {

        return BatchNormalizationLayer {
            training: true,
            eps: eps,
            momentum: momentum,
            gamma: Array2::<f32>::ones(
                (1, dimensions)
            ),
            beta: Array2::<f32>::zeros(
                (1, dimensions)
            ),
            running_mean: Array2::<f32>::zeros(
                (1, dimensions)
            ),
            running_var: Array2::<f32>::ones(
                (1, dimensions)
            ),
            grad: None,
            output: None,
            input: None
        };
    }

    
}

impl Layer for BatchNormalizationLayer {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.input = Some(x.clone());
        // if it is not training, use the running mean and variance
        // let mut xmean = self.running_mean.clone();
        // let mut xvar = self.running_var.clone();

        // if self.training {
        //     xmean = x.mean_axis(Axis(0)).unwrap();
        //     xvar = x.var_axis(Axis(0), 0.0);
        // } 

        // return x;
        todo!()
    }

    fn parameters(&self) -> Vec<Array2<f32>>{
        return vec![self.gamma.clone(), self.beta.clone()];
    }
    
    fn grad(&mut self, previous_grad: &Array2<f32>) -> Vec<Array2<f32>> {
        todo!()
    }
    
    fn get_grads(&self) -> Vec<Array2<f32>> {
        todo!()
    }
    
    fn get_type(&self) -> crate::core::layer::LayerType {
        todo!()
    }
    
    fn reset_grads(&mut self) {
        self.grad = None;
    }
    
    fn get_output(&self) -> Option<Array2<f32>> {
        return self.output.clone();
    }

}