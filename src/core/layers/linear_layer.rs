use crate::core::layer::Layer;
use ndarray::Array2;
use ndarray_rand::{rand_distr::{Normal, Uniform}, RandomExt};


#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub bias: Option<Array2<f32>>,
    pub grad: Vec<Array2<f32>>,
    pub output: Option<Array2<f32>>,
    pub input: Option<Array2<f32>>,
}

impl LinearLayer {
    pub fn new(fan_in: usize, fan_out: usize, bias: bool) -> LinearLayer {
        let rand_array  = Array2::<f32>::random(
            (fan_in, fan_out), 
            Normal::new(0.0, 1.0).ok().unwrap()
        ) / (fan_in as f64).sqrt() as f32; // Kaiming initialization

        return LinearLayer {
            weights: rand_array,
            bias: if bias {
                Some(Array2::<f32>::zeros(
                    (1, fan_out)
                ))
            } else {
                None
            },
            grad: vec![],
            output: None,
            input: None
        }
    }

}


impl Layer for LinearLayer {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.input = Some(x.clone());
        let mut out = x.dot(&self.weights);
        if self.bias.is_some() {
            out += self.bias.as_ref().unwrap();
        }
        self.output = Some(out.clone());
        return out;
    }


    fn parameters(&self) -> Vec<Array2<f32>>{
        return if self.bias.is_some() {
            vec![self.weights.clone(), self.bias.as_ref().unwrap().clone()]
        } else {
            vec![self.weights.clone()]
        }
    }
    
    fn grad(&mut self, previous_grad: &Array2<f32>) -> Vec<Array2<f32>> {
        
        let grad_weights = self.input.as_ref().unwrap().t().dot(previous_grad);

        let grad_input = previous_grad.dot(&self.weights.t());

        let grad_bias = if self.bias.is_some() {
            Some(previous_grad.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)))
        } else {
            None
        };
        
        let grads = if self.bias.is_some() {
            vec![grad_input, grad_weights, grad_bias.unwrap() ]
        } else {
            vec![grad_input, grad_weights]
        };

        self.grad = grads.clone();

        return grads;

    }

    fn get_grads(&self) -> Vec<Array2<f32>> {
        return self.grad.clone();
    }
    
    fn get_type(&self) -> crate::core::layer::LayerType {
        return crate::core::layer::LayerType::Linear;
    }
    
    fn reset_grads(&mut self) {
        self.grad = vec![];
    }
    
    fn get_output(&self) -> Option<Array2<f32>> {
        return self.output.clone();
    }
}

// Reference:
// https://arxiv.org/pdf/1502.01852v1.pdf