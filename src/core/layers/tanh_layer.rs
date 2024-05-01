use crate::core::layer::Layer;
use ndarray::Array2;

#[derive(Debug)]
pub struct TanhLayer {
    pub output: Option<Array2<f32>>,
    pub grad: Vec<Array2<f32>>,
    pub input: Option<Array2<f32>>
}

impl TanhLayer {
    pub fn new() -> TanhLayer {
        return TanhLayer {
            output: None,
            grad: vec![],
            input: None
        };
    }

}


impl Layer for TanhLayer {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.input = Some(x.clone());
        let out = x.mapv(|x| x.tanh());
        self.output = Some(out.clone());
        return out;

    }


    fn parameters(&self) -> Vec<Array2<f32>>{
        return  vec![]
    }
    
    fn grad(&mut self, previous_grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let grad = previous_grad * (1.0 - self.output.as_ref().unwrap().mapv(|x| x.powi(2)));
        self.grad.push(grad.clone());
        return vec![grad];
    }
    
    fn get_grads(&self) -> Vec<Array2<f32>> {
        return self.grad.clone();
    }
    
    fn get_type(&self) -> crate::core::layer::LayerType {
        return crate::core::layer::LayerType::Tanh
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