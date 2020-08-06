<?php

namespace PHPW2V\SoftmaxApproximators;

use PHPW2V\Word2Vec;
use PHPW2V\Trees\Heap;
use Tensor\Vector;
use InvalidArgumentException;

class HierarchicalSoftmax implements SoftmaxApproximator
{
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
     * Create sampling structure used in pair train approximation.
     *
     * @param Word2Vec $word2vec
     * @throws \InvalidArgumentException
     */
    public function structureSampling(Word2Vec $word2vec) : void
    {
        if (!$word2vec instanceof Word2Vec) {
            throw new InvalidArgumentException('Hierarchical Softmax requires a valid Word2Vec object to create binary tree.');
        }

        $this->vocab = $word2vec->vocab();
        $vocabCount = $word2vec->vocabCount();

        $heap = $this->buildHeap($this->vocab);
        $maxDepth = 0;
        $stack = [[$heap[0], [], []]];

        while ($stack) {
            $stackItem = array_pop($stack);

            if (empty($stackItem)) {
                break;
            }

            $points = $stackItem[2];
            $codes = $stackItem[1];
            $node = $stackItem[0];

            if ($node['index'] < $vocabCount) {
                $this->vocab[$node['word']]['code'] = Vector::quick($codes);
                $this->vocab[$node['word']]['point'] = $points;

                $maxDepth = max(count($codes), $maxDepth);
            } else {
                $points[] = ($node['index'] - $vocabCount);
                $codeLeft = $codeRight = $codes;

                $codeLeft[] = 0;
                $codeRight[] = 1;

                $stack[] = [$node['left'], $codeLeft, $points];
                $stack[] = [$node['right'], $codeRight, $points];
            }
        }
    }

    /**
     * Return the word indices used to update the hidden layer.
     *
     * @param mixed[] $predictWord
     * @return int[]
     */
    public function wordIndices(array $predictWord) : array
    {
        return $this->vocab[$predictWord['word']]['point'];
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
        return $fa->addVector($this->vocab[$predictWord]['code'])->negate()->addScalar(1)->multiplyScalar($alpha);
    }

    /**
     * Build a heap queue, prioritizing each word's respective word count, to initialize the binary tree.
     * Vocabulary array must include count index and value for each word.
     *
     * @param array[] $vocabulary
     * @return array[] $heap
     */
    private function buildHeap(array $vocabulary) : array
    {
        $heap = new Heap($vocabulary);
        $maxRange = (count($vocabulary) - 2);

        for ($i = 0; $i <= $maxRange; ++$i) {
            $min1 = $heap->heappop();
            $min2 = $heap->heappop();

            if (!empty($min1) && !empty($min2)) {
                $new_item = [
                    'count' => ($min1['count'] + $min2['count']),
                    'index' => ($i + (count($vocabulary))),
                    'left' => $min1,
                    'right' => $min2
                ];

                $heap->heappush($new_item);
            }
        }

        return $heap->heap();
    }
}
