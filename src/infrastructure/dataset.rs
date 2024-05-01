use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView, Axis, Dim, OwnedRepr};
use ndarray::s;use crate::core::dataset::{DatabaseBuilderErrors, DatasetAbstract, DatasetComponent};
use crate::core::vocabulary;

pub struct DatasetBuilder {
    pub words: Vec<String>,
    pub splits_params: (f64, f64, f64),
}


impl DatasetAbstract for DatasetBuilder {
    fn load(
        context_length: usize,
        vocab: &vocabulary::Vocabulary,
        splits_params: (f64, f64, f64)
    ) -> Result<crate::core::dataset::Dataset, crate::core::dataset::DatabaseBuilderErrors> {
        // Split the words into training, validation and test sets
        let sets = split_words(&vocab.words, splits_params);
        
        // Build the dataset components
        let (x_tr, y_tr) = match build(context_length, &vocab, &sets.0) {
            Ok((x, y)) => (x, y),
            Err(e) => return Err(e),
        };

        let (x_val, y_val) = match build(context_length, &vocab, &sets.1) {
            Ok((x, y)) => (x, y),
            Err(e) => return Err(e),
        };

        let (x_test, y_test) = match build(context_length, &vocab, &sets.2) {
            Ok((x, y)) => (x, y),
            Err(e) => return Err(e),
        };

        return Ok(crate::core::dataset::Dataset {
            x_tr: x_tr,
            y_tr: y_tr,
            x_val: x_val,
            y_val: y_val,
            x_test: x_test,
            y_test: y_test,
        });
    }

}


fn split_words(
    words: &Vec<String>,
    splits_params: (f64, f64, f64),
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut training: Vec<String> = Vec::new();
    let mut validation: Vec<String> = Vec::new();
    let mut test: Vec<String> = Vec::new();

    let training_size = (words.len() as f64 * splits_params.0).round() as usize;
    let validation_size = (words.len() as f64 * splits_params.1).round() as usize;

    for i in 0..words.len() {
        if i < training_size {
            training.push(words[i].to_owned());
        } else if i < training_size + validation_size {
            validation.push(words[i].to_owned());
        } else {
            test.push(words[i].to_owned());
        }
    }

    return (training, validation, test);
}


fn build(
    context_length: usize,
    vocab: &vocabulary::Vocabulary,
    words: &Vec<String>
) -> Result<(DatasetComponent, DatasetComponent), DatabaseBuilderErrors>{
    // Create empty sets
    let mut x = Array2::<f32>::zeros((0, context_length));
    let mut y = Array2::<f32>::zeros((1, 0));

    for (_, word) in words.iter().enumerate() {
        // Create a context array, initialized with only empty tokens
        let mut context: Array1<f32> = Array1::<f32>::zeros(context_length);

        // for each token in the processed text
        for c in word.chars() {
            // Get the index of the token in the vocabulary
            let ix = match vocab.stoi.get(&c.to_string()) {
                Some(ix) => ix,
                None => return Err(DatabaseBuilderErrors::UnmatchingToken),
            };
            
            // Broadcast the context array to the shape (1, context_length)
            let ctx_broadcasted = ArrayView::from(&context).into_shape((1, context_length)).unwrap();
            
            // Append the context array to the x set
            let _ = match x.append(Axis(0), ctx_broadcasted) {
                Ok(_) => (),
                Err(_) => return Err(DatabaseBuilderErrors::AppendingError),
            };
            
            // Append the respective most probable token to the y set
            let _ = match y.push_column(ArrayView::from(&[*ix as f32])) {
                Ok(_) => (),
                Err(_) => return Err(DatabaseBuilderErrors::AppendingError),
            };

            // Shift the context array to the right and append the new token
            context = context.slice(s![1..]).to_owned();
            
            let mut new_context = context.to_vec();
            
            new_context.push(*ix as f32);

            context = Array::from(new_context);
        
        };
    }

    return Ok((x, y));
}