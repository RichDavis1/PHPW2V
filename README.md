# PHPW2V
A PHP implementation of Word2Vec, a popular word embedding algorithm created by Tomas Mikolov and popularized by Radim Řehůřek &amp; Peter Sojka with the Gensim Python library

## Installation
Install PHPW2V into your project using [Composer](https://getcomposer.org/):
```sh
$ composer require rich-davis1/phpw2v
```

### Requirements
- [PHP](https://php.net/manual/en/install.php) 7.4 or above



## Using PHPW2v


### Step 1: Require Vendor autoload and import PHPW2V at the top of your file

```
<?php

require __DIR__ . '/vendor/autoload.php';

use PHPW2V\Word2Vec;
use PHPW2V\SoftmaxApproximators\NegativeSampling;
```


### Step 2: Prepare an array of sentences

```
$sentences = [
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
];

```


### Step 3: Train your model & save it for use later

```
$dimensions     = 150; //vector dimension size
$sampling       = new NegativeSampling; //Softmax Approximator
$minWordCount   = 2; //minimum word count
$alpha          = .05; //the learning rate
$window         = 3; //window for skip-gram
$epochs         = 500; //how many epochs to run
$subsample      = 0.05; //the subsampling rate


$word2vec = new Word2Vec($dimensions, $sampling, $window, $subsample,  $alpha, $epochs, $minWordCount);
$word2vec->train($sentences);
$word2vec->save('my_word2vec_model');
```


### Step 4: Load your previously trained model and find the most similar words 
```
$word2vec = new Word2Vec();
$word2vec = $word2vec->load('my_word2vec_model');

$mostSimilar = $word2vec->mostSimilar(['dog']);
```

Which results in:
```
Array
(
    [fox] => 0.65303660275952
    [pug] => 0.63475600376409
    [you] => 0.63469270773687
    [cat] => 0.28333476473645
    [are] => 0.0086017358485732
    [ran] => -0.016116842526914
    [the] => -0.068253396295047
    [runs] => -0.11967150816883
    [fast] => -0.12999690227979
)
```


### Step 5: Find similar words with both positive and negative contexts
```
$mostSimilar = $word2vec->mostSimilar(['dog'], ['cat']);
```


### Step 6: Get the word embedding of a word to be used in other NLP projects
```
$wordEmbedding = $word2vec->wordVec('dog');
```


