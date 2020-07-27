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
$dimensions     = 100; //vector dimension size
$sampling       = 'neg'; //accepts neg or hs
$min_word_count = 1; //minimum word count
$alpha          = .05; //the learning rate
$window         = 2; //window for skip-gram
$epochs         = 1000; //how many epochs to run
$subsample      = 0; //the subsampling rate


$word2vec = new Word2Vec($dimensions, $sampling, $window, $subsample,  $alpha, $epochs, $min_word_count);
$word2vec->train($sentences);
$word2vec->save('my_word2vec_model');
```


### Step 4: Load your previously trained model and find the most similar words 
```
$word2vec = new Word2Vec();
$word2vec = $word2vec->load('my_word2vec_model');

$most_similar = $word2vec->mostSimilar(['dog']);
```

Which results in:
```
Array
(
    [fox] => 0.70975123389235
    [pug] => 0.6864516587575
    [only] => 0.61312080700673
    [a] => 0.57297257749209
    [is] => 0.50647272674305
    [ran] => 0.39611266149472
    [yourself] => 0.3389617422606
    [to] => 0.33127041727032
    [jogged] => 0.30974688889277
)
```


### Step 5: Find similar words with both positive and negative contexts
```
$most_similar = $word2vec->mostSimilar(['dog'], ['cat']);
```


### Step 6: Get the word embedding of a word to be used in other NLP projects
```
$word_embedding = $word2vec->wordVec('dog');
```


