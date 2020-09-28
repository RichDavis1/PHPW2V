<?php

namespace PHPW2V\Trees;

use InvalidArgumentException;

/**
 * Heap
 *
 * A heap is a specialized tree-based data structure that satisfies the heap property:
 * in a max heap, for any given node C, if P is a parent node of C, then the key of P
 * is greater than or equal to the key of C. In a min heap, the key of P is less than or equal to the key of C.
 *
 * References:
 * [1] Black (ed.), Paul E. (2004-12-14)
 * <https://xlinux.nist.gov/dads/HTML/heap.html>
 *
 * @category    Machine Learning
 * @package     RubixML
 * @author      Rich Davis
 */
class Heap
{
    /**
     * An array of words from a corpus arranged in a manner that maintains the heap invariant.
     *
     * @var mixed[]
     */
    protected $heap;

    /**
     * @param array[] $x
     */
    public function __construct(array $x)
    {
        $this->heap = $this->heapify($x);
    }

    /**
     * Returns the heap.
     *
     * @return array[]
     */
    public function heap() : array
    {
        return $this->heap;
    }

    /**
     * Pops and returns the smallest item in the heap.
     *
     * @return mixed[]|null
     */
    public function heappop() : ?array
    {
        if (!is_array($this->heap) or empty($this->heap)) {
            return null;
        }

        $last_element = array_pop($this->heap);

        if (!is_array($this->heap) or empty($this->heap)) {
            return $last_element;
        }

        $return_item = $this->heap[0];

        $this->heap[0] = $last_element;
        $this->heap = $this->siftup($this->heap, 0);

        return $return_item;
    }

    /**
     * Pushes the value of the item in the heap.
     *
     * @param mixed[] $item
     * @return mixed[]
     */
    public function heappush(array $item) : array
    {
        $this->heap[] = $item;
        $new_count = (count($this->heap) - 1);

        $this->siftdown(0, $new_count);

        return $this->heap;
    }

    /**
     * Finds the best fit for the new item & updates heap while maintaining heap invariant.
     *
     * @param int $start_pos
     * @param int $i
     */
    private function siftdown(int $start_pos, int $i) : void
    {
        $temp_item = $this->heap[$i];

        while ($i > $start_pos) {
            $parent_pos = ($i - 1) >> 1;
            $parent = $this->heap[$parent_pos];

            if (($temp_item['count'] < $parent['count'])) {
                $this->heap[$i] = $parent;
                $i = $parent_pos;

                continue;
            }

            break;
        }

        $this->heap[$i] = $temp_item;
    }

    /**
     * Creates a heap at the provided index while maintaining heap invariant.
     *
     * @param mixed[] $x
     * @param int $i
     * @return mixed[]
     */
    private function siftup(array $x, int $i) : array
    {
        $end_pos = count($x);
        $start_pos = $i;
        $new_item = $this->heap[$i];
        $child_pos = (2 * $i) + 1;

        while ($child_pos < $end_pos) {
            $right_pos = $child_pos + 1;

            if (empty($this->heap[$right_pos]['count'])) {
                $right_heap = 0;
            } else {
                $right_heap = $this->heap[$right_pos]['count'];
            }

            if ($right_pos < $end_pos and ($this->heap[$child_pos]['count'] >= $right_heap)) {
                $child_pos = $right_pos;
            }

            $this->heap[$i] = $this->heap[$child_pos];
            $i = $child_pos;
            $child_pos = (2 * $i) + 1;
        }

        $this->heap[$i] = $new_item;
        $this->siftdown($start_pos, $i);

        return $this->heap;
    }

    /**
     * Converts an array into a heap.
     *
     * @param array[] $x
     * @return mixed[]
     */
    private function heapify(array $x) : array
    {
        $this->heap = array_values($x);
        $n = count($this->heap);

        $maxRange = (($n / 2) - 1);

        for ($i = $maxRange; $i >= 0; --$i) {
            $this->heap = $this->siftup($this->heap, ((int) $i));
        }

        return $this->heap;
    }
}
