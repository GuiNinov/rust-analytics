
use std::vec;

use ndarray::{Array2, Axis};
use ndarray_rand::{rand_distr::{Normal, Uniform}, RandomExt};

use crate::core::layer::{Layer, LayerType};



#[derive(Debug)]
pub struct LayerNormalization1DLayer {
    training: bool,
    dimensions: usize,
    eps: f32,
    momentum: f32,
    gamma: Array2<f32>,
    beta: Array2<f32>,
    grad: Vec<Array2<f32>>,    
    output: Option<Array2<f32>>,
    input: Option<Array2<f32>>,
}

impl LayerNormalization1DLayer {
    pub fn new(dimensions: usize, eps: f32, momentum: f32) -> LayerNormalization1DLayer {

        return LayerNormalization1DLayer {
            training: true,
            eps: eps,
            momentum: momentum,
            gamma: Array2::<f32>::ones(
                (1, dimensions)
            ),
            beta: Array2::<f32>::zeros(
                (1, dimensions)
            ),
            dimensions: dimensions,
            grad: vec![],
            output: None,
            input: None
        };
    }
    
    
}

impl Layer for LayerNormalization1DLayer {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {

        self.input = Some(x.clone());

        let xmean = x.mean_axis(Axis(1)).unwrap();
        let xvar = x.var_axis(Axis(1), 0.0);
    
        let normalized_var = xvar + self.eps;
        let sqrt_array = normalized_var.mapv(f32::sqrt);
    
        // Ensure xmean has the same number of columns as x for broadcasting
        let binding_mean = &xmean.insert_axis(Axis(1));
        let formatted_xmean = &binding_mean.broadcast(x.dim()).unwrap();

        let binding_sqrt = &sqrt_array.insert_axis(Axis(1));
        let formatted_sqrt = &binding_sqrt.broadcast(x.dim()).unwrap();

        let raw_normalization = (x - formatted_xmean) / formatted_sqrt;

        let out = &self.gamma * raw_normalization + &self.beta;
        
        
        self.output = Some(out);

        return self.output.clone().unwrap();
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        let params = vec![
            self.gamma.clone(),
            self.beta.clone()
        ];

        return  params;
    }

    fn grad(&mut self, previous_grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let x = self.input.clone().unwrap();

        let xmean = x.mean_axis(Axis(1)).unwrap();
        let xvar = x.var_axis(Axis(1), 0.0);
        
        let normalized_var = xvar + self.eps;
        let sqrt_array = normalized_var.mapv(f32::sqrt);
        
        let binding_mean = &xmean.insert_axis(Axis(1));
        let formatted_xmean = &binding_mean.broadcast(x.dim()).unwrap();
        let binding_sqrt = &sqrt_array.insert_axis(Axis(1));
        let formatted_sqrt = &binding_sqrt.broadcast(x.dim()).unwrap();
        
        let raw_normalization = (x - formatted_xmean) / formatted_sqrt;
        
        // dgamma
        let dgamma_values = (previous_grad * &raw_normalization).sum_axis(Axis(0));        
        // Broadcast dgamma_values to the same shape as gamma
        let dgamma_2d = dgamma_values.clone().insert_axis(Axis(0));
        let dgamma = dgamma_2d.broadcast(self.gamma.dim()).unwrap().to_owned();
        
        // dbeta
        let dbeta_values = previous_grad.sum_axis(Axis(0));
        // Broadcast dbeta_values to the same shape as beta
        let dbeta = dbeta_values.clone().insert_axis(Axis(0)).broadcast(self.beta.dim()).unwrap().to_owned();
        // dx 
        let n = previous_grad.shape()[0] as f32;
        let dx = &self.gamma / (formatted_sqrt*n) * (n * previous_grad - previous_grad.sum_axis(Axis(0)) - n / (n - 1.0) * &raw_normalization * &(previous_grad * &raw_normalization).sum_axis(Axis(0)));        let out = vec![
            dx,
            dgamma,
            dbeta
        ];

        self.grad = out.clone();
        
        return out.clone();
    }

    fn get_grads(&self) -> Vec<Array2<f32>> {
        return self.grad.clone();
    }

    fn reset_grads(&mut self) {
        self.grad = vec![];
    }

    fn get_type(&self) -> crate::core::layer::LayerType {
        return crate::core::layer::LayerType::LayerNormalization1DLayer;
    }
    
    fn get_output(&self) -> Option<Array2<f32>> {
        return self.output.clone();
    }
}