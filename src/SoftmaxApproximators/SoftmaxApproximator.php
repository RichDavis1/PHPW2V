<?php

namespace PHPW2V\SoftmaxApproximators;

use PHPW2V\Word2Vec;
use Tensor\Vector;

interface SoftmaxApproximator
{
    /**
     * Create sampling structure used in pair train approximation.
     *
     * @param \PHPW2V\Word2Vec $word2vec
     * @throws \InvalidArgumentException
     */
    public function structureSampling(Word2Vec $word2vec) : void;

    /**
     * Return the word indices used to update the hidden layer.
     *
     * @param mixed[] $predictWord
     * @return int[]
     */
    public function wordIndices(array $predictWord) : array;

    /**
     * Calculate the gradient descent.
     *
     * @param Vector $fa
     * @param string $predictWord
     * @param float $alpha
     * @return Vector
     */
    public function gradientDescent(Vector $fa, string $predictWord, float $alpha) : Vector;
}
