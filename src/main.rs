mod infrastructure;
mod core;


use ndarray::{Array, Array1, Array2, ArrayView, Axis};
use ndarray::s;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use core::layers::layer_norm_1d::LayerNormalization1DLayer;
use std::env;
use std::env::VarError;
use dotenv::dotenv;

use core::vocabulary::VocabularyAbstract;
use infrastructure::vocabulary::VocabularyService;
use crate::core::dataset::DatasetAbstract;
use crate::core::layer::{Layer, Layers};
use crate::core::layers::batch_norm_layer::BatchNormalizationLayer;
use crate::core::layers::linear_layer::LinearLayer;
use crate::core::layers::tanh_layer::TanhLayer;
use crate::core::model::Model;


fn main() {
    // Loading environment variables
    dotenv().ok();
    let file_path: String = env::var("FILE_PATH").ok().unwrap();
    let context_length: usize = env::var("CONTEXT_LENGTH").ok().unwrap().parse::<usize>().unwrap();
    let dimensions: usize = env::var("DIMENSIONS").ok().unwrap().parse::<usize>().unwrap();
    let neurons: usize = env::var("NEURONS").ok().unwrap().parse::<usize>().unwrap();
    let max_steps: usize = env::var("MAX_STEPS").ok().unwrap().parse::<usize>().unwrap();
    let batch_size: usize = env::var("BATCH_SIZE").ok().unwrap().parse::<usize>().unwrap();
    let learning_rate: f32 = env::var("LEARNING_RATE").ok().unwrap().parse::<f32>().unwrap();

    let vocab_service = VocabularyService::new(file_path);
    let vocab = match vocab_service.find() {
        Ok(vocab) => vocab,
        _ => {
            panic!("Vocabulary not found");
        }
    };

    let dataset =  match infrastructure::dataset::DatasetBuilder::load(
        context_length,
        &vocab,
        (0.8, 0.1, 0.1),
    )
     {
        Ok(dataset) => dataset,
        _ => {
            panic!("Dataset not found");
        }
    };

    let embeddings = Array2::<f32>::random(
        (vocab.vocab_size, dimensions), 
Normal::new(0.0, 1.0).ok().unwrap()
    );

    let eps = 1e-7;
    let momentum = 0.9;

    // type layers = LinearLayer || BatchNormalizationLayer || TanhLayer;

    let layers:Vec<Box<dyn Layer>>  = vec![
        Box::new(LinearLayer::new(dimensions * context_length, neurons, false)), 
        Box::new(LayerNormalization1DLayer::new(neurons, eps, momentum)), 
        Box::new(TanhLayer::new()),
        //
        Box::new(LinearLayer::new(neurons, neurons, false)), 
        Box::new(LayerNormalization1DLayer::new(neurons, eps, momentum)), 
        Box::new(TanhLayer::new()),
        //
        
        Box::new(LinearLayer::new(neurons, neurons, false)), 
        Box::new(LayerNormalization1DLayer::new(neurons, eps, momentum)), 
        Box::new(TanhLayer::new()),
        //
        Box::new(LinearLayer::new(neurons, vocab.vocab_size, false)), 
        Box::new(LayerNormalization1DLayer::new(vocab.vocab_size, eps, momentum)),
    ];

    let mut model: Model = Model::new(
        layers,
        embeddings,
        dataset,
        context_length,
        dimensions,
        max_steps,
        learning_rate,
        batch_size,
    );

    // println!("{:?}", model.layers.len());
    model.train();

    model.stats();

    // println!("{:?}", l);

    // println!("{:?}", embeddings);


}


