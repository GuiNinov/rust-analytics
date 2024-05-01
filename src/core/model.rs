use std::{collections::HashMap, vec};

use ndarray::{concatenate, s, Array2, Array3, Axis, Zip};
use ndarray_rand::{rand_distr::num_traits::ToPrimitive, RandomExt};

use super::{dataset::Dataset, layer::Layer};

use rand::prelude::*;


pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    embeddings: Array2<f32>,
    dataset: Dataset,
    max_steps: usize,
    batch_size: usize,
    dimensions: usize,
    context_length: usize,
    learning_rate: f32,
    lossi: Vec<f32>,
}

impl Model {
    pub fn new(
        layers: Vec<Box<dyn Layer>>, 
        embeddings: Array2<f32>,
        dataset: Dataset,
        context_length: usize,
        dimensions: usize,
        max_steps: usize,
        learning_rate: f32,
        batch_size: usize,
    ) -> Model {

        let mut parameters: Vec<Array2<f32>> = vec![];

        for layer in layers.iter() {
            parameters.extend(layer.parameters());
        }

        parameters.push(embeddings.clone());

        let total_parameters = parameters.iter().fold(0, |acc, p| {
            acc + p.len()
        });

        println!("Total parameters: {:?}", total_parameters);

        return Model {
            layers: layers,
            embeddings: embeddings,
            dataset: dataset,
            dimensions: dimensions,
            context_length: context_length,
            batch_size: batch_size,
            max_steps: max_steps,
            learning_rate: learning_rate,
            lossi: vec![],
        };
    }

    fn reset_grads(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_grads();
        }
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        let mut parameters: Vec<Array2<f32>> = vec![self.embeddings.clone()];

        for layer in self.layers.iter() {
            layer.parameters().iter().for_each(|p| {
                parameters.push(p.clone());
            });
        }

        return parameters;
    }

    pub fn train(&mut self) {
        
        self.reset_grads();

        for step in 0..self.max_steps {

        // Creating the mini-batch ---------------------   
            let (x_batch, y_batch) = self.generate_batch();
        // --------------------------------------------

        // Embedding lookup ----------------------------
            let x_batch_reshaped: Array3<f32> = x_batch.clone().into_shape((self.batch_size, self.context_length, 1)).unwrap();
            
            let mut embedding_values = Array3::<f32>::zeros((self.batch_size, self.context_length, self.dimensions));
            
            for (i, mut embedding_values_row) in embedding_values.axis_iter_mut(Axis(0)).enumerate() {
                for (j, mut embedding_values_col) in embedding_values_row.axis_iter_mut(Axis(0)).enumerate() {
                    let token_index = x_batch_reshaped[(i, j, 0)] as usize; // Access token index from x_batch
                    let embedding_row = self.embeddings.slice(s![token_index, ..]); // Get corresponding row from embedding
                    embedding_values_col.assign(&embedding_row); // Assign the row to the result
                }
            }
            
            let mut concatenated_result: Array2<f32> = embedding_values.into_shape((32, 30)).unwrap();
        // --------------------------------------------

        // Forward pass -------------------------------
            for layer in self.layers.iter_mut() {
                concatenated_result = layer.forward(&concatenated_result);
            }
            let loss = self.loss(&concatenated_result, &y_batch);
        // --------------------------------------------

        // Backward pass ------------------------------

            // - dloss/dlogits = d[ - log(SOFTMAX) ] / dlogits = SOFTMAX - 1 or SOFTMAX
            let grad_logits = self.grad_logits(&concatenated_result, &y_batch);

                
            let mut grads: Vec<Array2<f32>> = Vec::new();
            let mut grad_params: Vec<Array2<f32>> = Vec::new();
            
            // gradient descent
            let layers_amount = self.layers.len();
            for i in 0..layers_amount {      
                let previous_layer_grads = if grads.len() != 0 {
                    self.layers[layers_amount - i ].get_grads()
                } else {
                    vec![grad_logits.clone()]
                };

                let layer_grads = self.layers[layers_amount - i - 1].grad(&previous_layer_grads[0]);
                
                grads = layer_grads.clone();
                
                match self.layers[layers_amount - i - 1].get_type() {
                    crate::core::layer::LayerType::Linear => {
                        grad_params.push(layer_grads[1].clone());
                        if layer_grads.len() == 3 {
                            grad_params.push(layer_grads[2].clone())
                        }
                    },
                    crate::core::layer::LayerType::LayerNormalization1DLayer => {
                        grad_params.push(layer_grads[1].clone());
                        grad_params.push(layer_grads[2].clone());
                    },
                    crate::core::layer::LayerType::BatchNormalization => todo!(),
                    crate::core::layer::LayerType::Tanh => {
                        continue;
                    },
                    _ => continue,
                }
            }

            // embedding gradient
            let first_layer_grad = self.layers[0].get_grads()[0].clone();

            let first_layer_grad_reshaped: Array3<f32> = first_layer_grad.into_shape((self.batch_size, self.context_length, self.dimensions)).unwrap();

            let mut grad_embeddings = Array2::<f32>::zeros((self.embeddings.shape()[0], self.embeddings.shape()[1]));

            for k in 0..x_batch.shape()[0]{
                for j in 0..x_batch.shape()[1]{
                    let token_index = x_batch[(k, j)] as usize;
                    let values = first_layer_grad_reshaped.slice(s![k, j, ..]);
                    grad_embeddings.row_mut(token_index).assign(&values);
                }
            }

            grad_params.push(grad_embeddings);
        // --------------------------------------------

        // Update parameters --------------------------
            let lr = self.learning_rate;

            let mut parameters = self.parameters();

            for i in 0..parameters.len() {
                let learned_parameters = -lr * grad_params[grad_params.len() - i -1].clone();                
                parameters[i] = &parameters[i] + &learned_parameters;
            }
        // --------------------------------------------

        // Track statistics ---------------------------
            self.lossi.push(loss);
            if step % 1000 == 0 {
                println!("Step: {:?}/{:?} -> Error: {:?}", step, self.max_steps, loss);
            }
        // --------------------------------------------
        }
        
        
    }

    fn loss(&self, logits: &Array2<f32>, y: &Array2<f32>) -> f32 {
        // Find the maximum values along axis 1
        let mut max_indices = Vec::new();
        for row in logits.axis_iter(Axis(0)) {
            let mut max_index = 0;
            let mut max_val = row[0];
            for (i, &val) in row.iter().enumerate().skip(1) {
                if val > max_val {
                    max_val = val;
                    max_index = i;
                }
            }
            max_indices.push(max_index);
        }

        // Collect the maximum values corresponding to max_indices
        let logit_maxes: Vec<f32> = max_indices
            .iter()
            .enumerate()
            .map(|(i, &index)| logits[[i, index]])
            .collect();

        // Subtract the maximum values for numerical stability
        let mut norm_logits = logits.clone();
        for (i, mut row) in norm_logits.axis_iter_mut(Axis(0)).enumerate() {
            let max_val = logit_maxes[i];
            for val in row.iter_mut() {
                *val -= max_val;
            }
        }

        // Compute the counts, counts_sum, and counts_sum_inv
        let counts = norm_logits.mapv(f32::exp);
        let counts_sum = counts.sum_axis(Axis(1)).insert_axis(Axis(1));
        let counts_sum_inv = counts_sum.mapv(|x| 1.0 / x);

        // Compute the probabilities
        let mut probs = &counts * &counts_sum_inv;

        // Compute the negative log likelihood
        let mut likelihoods = vec![];
        for (i, mut row) in probs.axis_iter_mut(Axis(0)).enumerate() {
            let y_val = y[[0, i]];
            likelihoods.push(
                row[y_val as usize].ln()
            )
            // row *= y_val;
        }
        
        let loss = - likelihoods.iter().sum::<f32>() / likelihoods.len() as f32;
        
    
        return loss;
    }

    fn grad_logits(&self, logits: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
        
        // exp each element of logits
        let exp_logits = logits.mapv(f32::exp);

        // sum of exp_logits along axis 1
        let sum_exp_logits = exp_logits.sum_axis(Axis(1)).insert_axis(Axis(1));

        // Compute the softmax probabilities
        let mut grad_logits = &exp_logits / &sum_exp_logits;

        // iterate grad_logits searching for the correct y value
        for (i, mut row) in grad_logits.axis_iter_mut(Axis(0)).enumerate() {
            let y_val = y[[0, i]];
            row[y_val as usize] -= 1.0;
        }

        return grad_logits;
    }

    fn generate_batch(&self) -> (Array2<f32>, Array2<f32>){
        // --- Create a random number generator
        let mut rng = thread_rng();

        // --- Create a 2D array of random integers
        let mut batch_factor = Array2::zeros((self.batch_size, 1));

        batch_factor.mapv_inplace(|_| rng.gen_range(0..self.dataset.x_tr.shape()[0])); // Adjust range as needed

        let x_tr = &self.dataset.x_tr;
        let y_tr = &self.dataset.y_tr;

        // filter x_tr and y_tr based on batch_factor;

        let selected_x_rows = batch_factor
        .iter()
        .map(|&idx| x_tr.index_axis(Axis(0), idx).to_owned()) // Convert to owned array
        .collect::<Vec<_>>();

        let selected_y_rows = batch_factor
        .iter()
        .map(|&idx| y_tr.index_axis(Axis(1), idx).to_owned()) // Select columns
        .collect::<Vec<_>>();


        let mini_batch_x = Array2::from_shape_fn(
            (self.batch_size, x_tr.shape()[1]),
            |(i, j)| selected_x_rows[i][j], // Access individual elements
        );
        
        let mini_batch_y = Array2::from_shape_fn(
            (1, self.batch_size), // Swap shape to match desired shape [1, 32]
            |(i, j)| selected_y_rows[j][i], // Swap indices to transpose the matrix
        );
        
        return (mini_batch_x, mini_batch_y);
    }

    pub fn stats(&mut self) {
        let mut parameters: Vec<Array2<f32>> = vec![self.embeddings.clone()];
        for layer in self.layers.iter() {
            parameters.extend(layer.parameters());
        }

        let total_parameters = parameters.iter().fold(0, |acc, p| {
            acc + p.len()
        });

        println!("Total parameters: {:?}", total_parameters);
        println!("Final loss: {:?}", self.lossi.last().unwrap());       
    }

    pub fn exec(&mut self) {
        todo!()
    }

}