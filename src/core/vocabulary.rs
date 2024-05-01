use std::collections::HashMap;

pub enum VocabularyErrors {
    FileNotFound,
    InternalServerError,
    ProviderRequestError,
}


#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub chars: Vec<char>,
    pub stoi: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
    pub vocab_size: usize,
    pub words: Vec<String>,
}

pub trait VocabularyAbstract {
    fn find(&self) -> Result<Vocabulary, VocabularyErrors>;

    fn copy(&self) -> Vocabulary;
}