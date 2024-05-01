use std::collections::HashMap;
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView, Axis, Dim, OwnedRepr};

use super::vocabulary;


pub type DatasetComponent = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;

#[derive(Debug)]
pub struct Dataset {
    pub x_tr: DatasetComponent,
    pub y_tr: DatasetComponent,
    pub x_val: DatasetComponent,
    pub  y_val: DatasetComponent,
    pub x_test: DatasetComponent,
    pub y_test: DatasetComponent,
}

#[derive(Debug)]
pub enum DatabaseBuilderErrors {
    InvalidContextLength,
    InvalidVocabulary,
    AppendingError,
    UnmatchingToken
}

pub trait DatasetAbstract {
    fn load(
        context_length: usize,
        vocabulary: &vocabulary::Vocabulary,
        split_params: (f64, f64, f64),
    ) -> Result<Dataset, DatabaseBuilderErrors>;
}