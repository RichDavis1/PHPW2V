<?php

namespace phpw2v;

class Heap{

	public function __construct($x){
		$this->heap = $x;
	}

	public function heappop(){
		if(!is_array($this->heap) || empty($this->heap)){
			return false;
		}

		$last_element = array_pop($this->heap);

		if(!is_array($this->heap) || empty($this->heap)){
			return $last_element;
		}

		$return_item = $this->heap[0];
		$this->heap[0] = $last_element;
		$this->heap = $this->siftup($this->heap, 0);

		return $return_item;
	}

	public function heappush($item){
		$this->heap[] = $item;
		$new_count    = (count($this->heap) - 1);

		$this->siftdown($this->heap, 0, $new_count); 

		return $this->heap;
	}

	public function siftdown($x, $start_pos, $i){
		$temp_item = $this->heap[$i];

		while( $i > $start_pos ){
			$parent_pos = ($i - 1) >> 1;
			$parent     = $this->heap[$parent_pos];

			if(($temp_item['count'] < $parent['count'])){
				$this->heap[$i] = $parent;
				$i = $parent_pos;

				continue;
			}

			break;
		}

		$this->heap[$i] = $temp_item;		
	}

	public function siftup($x, $i){
		$end_pos   = count($x);
		$start_pos = $i;
		$new_item  = $this->heap[$i];
		$child_pos = (2 * $i) + 1;

		while($child_pos < $end_pos){
			$right_pos = $child_pos + 1;

			if(empty($this->heap[$right_pos]['count'])){
				$right_heap = 0;
			}else{
				$right_heap = $this->heap[$right_pos]['count'];
			}
			
			if($right_pos < $end_pos && ($this->heap[$child_pos]['count'] >= $right_heap)) {
				$child_pos = $right_pos;
			}

			$this->heap[$i] = $this->heap[$child_pos];
			$i = $child_pos;
			$child_pos = (2 * $i) + 1;
		}

		$this->heap[$i] = $new_item; 
		$this->siftdown($this->heap, $start_pos, $i);

		return $this->heap;
	}

	public function heapify(){
		$this->heap = array_values($this->heap);
		$n = count($this->heap);		

		$max_range = ( ($n / 2) - 1 );

		foreach(array_reverse(range(0, $max_range)) as $i){
			$this->heap = $this->siftup($this->heap, $i);
		}

		return $this->heap;
	}


}