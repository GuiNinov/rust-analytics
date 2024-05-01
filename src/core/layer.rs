use ndarray::Array2;

use super::layers::{batch_norm_layer::BatchNormalizationLayer, layer_norm_1d::LayerNormalization1DLayer, linear_layer::LinearLayer, tanh_layer::TanhLayer};

pub trait Layer {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32>;
    fn parameters(&self) -> Vec<Array2<f32>>;
    fn grad(&mut self, previous_grad: &Array2<f32>) -> Vec<Array2<f32>>;
    fn get_grads(&self) -> Vec<Array2<f32>>;
    fn reset_grads(&mut self);
    fn get_type(&self) -> LayerType;
    fn get_output(&self) -> Option<Array2<f32>>;
}

#[derive(Debug)]
pub enum LayerType {
    Linear,
    BatchNormalization,
    LayerNormalization1DLayer,
    Tanh
}

#[repr(C)]
pub enum Layers {
    LinearLayer(LinearLayer),
    BatchNormalizationLayer(BatchNormalizationLayer),
    TanhLayer(TanhLayer),
    LayerNormalization1DLayer(LayerNormalization1DLayer)
}