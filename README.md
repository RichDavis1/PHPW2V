# PHPW2V
A PHP implementation of Word2Vec, a popular word embedding algorithm created by Tomas Mikolov and popularized by Radim Řehůřek &amp; Peter Sojka with the Gensim Python library

## Installation
Install PHPW2V into your project using [Composer](https://getcomposer.org/):
```sh
$ composer require rich-davis1/phpw2v
```

### Requirements
- [PHP](https://php.net/manual/en/install.php) 7.2 or above



## Using PHPW2v


### Step 1: Require Vendor autoload and import PHPW2V

```
<?php

require __DIR__ . '/vendor/autoload.php';

use phpw2v\Word2Vec;
```

### Step 2: Prepare an array of sentences

```
$sentences = array(
    'the fox runs fast',
    'the cat jogged fast',
    'the pug ran fast',
    'the cat runs fast',
    'the dog ran fast',
    'the pug runs fast',
    'the fox ran fast',
    'dogs are our link to paradise',
    'pets are humanizing',
    "a dog is the only thing on earth that loves you more than you love yourself",    
);

```


### Step 3: Train your model & save it for use later

```
$dimensions     = 100; //vector dimension size
$sampling       = 'neg'; //accepts neg or hs
$min_word_count = 2; //minimum word count
$alpha          = .05; //the learning rate
$window         = 3; //window for skip-gram
$epochs         = 100; //how many epochs to run
$subsample      = 0; //the subsampling rate


$word2vec = new Word2Vec($sampling, $window, $dimensions, $subsample,  $alpha, $epochs, $min_word_count);
$word2vec->train($sentences);
$word2vec->save('my_word2vec_model');
```

Which results in:
```
Array
(
    [pug] => 0.9122636145201
    [fox] => 0.91121772783449
    [cat] => 0.87139391851075
    [you] => 0.68319173725482
    [runs] => 0.29705252901269
    [ran] => 0.28222306054137
    [are] => -0.044915405981431
    [fast] => -0.11318860822446
    [the] => -0.98209420172572
)
```


### Step 4: Load your previously trained model and find the most similar words 
```
$word2vec = new Word2Vec();
$word2vec = $word2vec->load('my_word2vec_model');

$most_similar = $word2vec->most_similar(['dog']);
```

### Step 5: Find similar words with both positive and negative contexts
```
$most_similar = $word2vec->most_similar(['dog'], ['cat']);
```

### Step 6: Get the word embedding of a word to be used in other NLP projects
```
$word_embedding = $word2vec->wordVec('dog');
```


