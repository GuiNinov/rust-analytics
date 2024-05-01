use std::collections::HashMap;
use std::fs;

use crate::core::vocabulary::{
    VocabularyErrors,
    VocabularyAbstract,
    Vocabulary
};

pub struct VocabularyService {
    file_path: String,
}

impl VocabularyAbstract for VocabularyService  {
    fn find(&self) -> Result<Vocabulary, VocabularyErrors> {
        let read = fs::read_to_string(&self.file_path);

        let contents = match read {
            Ok(contents) => contents,
            Err(_) => return Result::Err(VocabularyErrors::FileNotFound),
        };

        // println!("Contents: {}", contents);
        let mut words: Vec<&str> = vec![];

        for line in contents.lines() {
            words.push(line);
        }

        let mut chars: Vec<char> = vec![];
        for word in words.iter() {
            for c in word.chars() {
                if !chars.contains(&c) {
                    chars.push(c);
                }
                continue;
            }
        }

        chars.sort();


        let mut stoi: HashMap<String, usize> = HashMap::new();
        for (i, s) in chars.iter().enumerate() {
            stoi.insert(s.to_string(), i + 1);
        }
        stoi.insert(".".to_string(), 0);

        let mut itos: HashMap<usize, String> = HashMap::new();
        for (s, i) in stoi.iter() {
            itos.insert(*i, s.to_string());
        }
        let vocab_size = itos.len();

        // println!("Chars: {:?}", chars);
        // println!("Stoi: {:?}", stoi);
        // println!("Itos: {:?}", itos);
        // println!("Vocab size: {}", vocab_size);
        // println!("Itos [5]: {:?}", itos.get(&5).unwrap());

        return Result::Ok(Vocabulary {
            chars: chars,
            stoi: stoi,
            itos: itos,
            vocab_size: vocab_size,
            words: words.iter().map(|s| s.to_string()).collect(),
        });
    }
    
    fn copy(&self) -> Vocabulary {
        return Vocabulary {
            chars: vec![],
            stoi: HashMap::new(),
            itos: HashMap::new(),
            vocab_size: 0,
            words: vec![],
        };
    }
}

impl VocabularyService {
    pub fn new(file_path: String) -> VocabularyService {
        VocabularyService {
            file_path: file_path,
        }
    }
}
