<?php

namespace PHPW2V\SoftmaxApproximators;

use PHPW2V\Word2Vec;
use Tensor\Vector;
use InvalidArgumentException;

class NegativeSampling implements SoftmaxApproximator
{
    /**
     * The negative sampling exponent.
     *
     * @var float
     */
    protected const NS_EXPONENT = 0.75;

    /**
     * An array containing each word in the corpus and it's respective index, count, and multiplier.
     *
     * @var array[]
     */
    protected $vocab = [];

    /**
     * The total number of unique words in the corpus.
     *
     * @var int
     */
    protected $vocabCount;

    /**
     * The cumulative distribution table for negative sampling.
     *
     * @var int[]
     */
    protected $cumTable = [];

    /**
     * The last digit in the cumulative distribution table.
     *
     * @var int
     */
    protected $endCumDigit;

    /**
     * The negative labels used in the cumulative distrubtion table.
     *
     * @var \Tensor\Vector
     */
    protected $negLabels;

    /**
     * Create sampling structure used in pair train approximation.
     *
     * @param \PHPW2V\Word2Vec $word2vec
     * @throws \InvalidArgumentException
     */
    public function structureSampling(Word2Vec $word2vec) : void
    {
        if (!$word2vec instanceof Word2Vec) {
            throw new InvalidArgumentException('Negative Sampling requires a valid Word2Vec object to create cumulative distribution table.');
        }

        $vocab = $word2vec->vocab();
        $vocabCount = $word2vec->vocabCount();
        $index2word = $word2vec->index2Word();
        $this->negLabels = Vector::quick([1, 0]);

        $domain = ((2 ** 31) - 1);
        $trainWordsPow = $cumulative = 0;
        $cumTable = array_fill(0, $vocabCount, 0);

        for ($i = 0; $i < $vocabCount; ++$i) {
            $trainWordsPow += ($vocab[$index2word[$i]]['count'] ** self::NS_EXPONENT);
        }

        for ($i = 0; $i < $vocabCount; ++$i) {
            $cumulative += ($vocab[$index2word[$i]]['count'] ** self::NS_EXPONENT);
            $cumTable[$i] = (int) round(($cumulative / $trainWordsPow) * $domain);
        }

        $this->cumTable = $cumTable;
        $this->endCumDigit = (int) end($cumTable);
    }

    /**
     * Return the word indices used to update the hidden layer.
     *
     * @param mixed[] $predictWord
     * @return int[]
     */
    public function wordIndices(array $predictWord) : array
    {
        $wordIndices = [$predictWord['index']];

        while (count($wordIndices) < 1 + 1) {
            $temp = $this->cumTable;
            $randInt = rand(0, $this->endCumDigit);
            $temp[] = $randInt;

            sort($temp);
            $w = array_search($randInt, $temp);

            if ($w !== $predictWord['index']) {
                $wordIndices[] = $w;
            }

            continue;
        }

        return $wordIndices;
    }

    /**
     * Calculate the gradient descent.
     *
     * @param Vector $fa
     * @param string $predictWord
     * @param float $alpha
     * @return Vector
     */
    public function gradientDescent(Vector $fa, string $predictWord, float $alpha) : Vector
    {
        return $this->negLabels->subtractVector($fa)->multiplyScalar($alpha);
    }
}
