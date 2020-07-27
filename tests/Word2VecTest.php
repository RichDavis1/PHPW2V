<?php

use PHPW2V\Word2Vec;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Embedders
 * @covers \Rubix\ML\Embedders\Word2Vec
 */
class Word2VecTest extends TestCase
{
    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const DATASET_SIZE = 2;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var \PHPW2V\Word2Vec
     */
    protected $model;

    /**
     * @var string[]
     */
    protected $sampleDataset;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->sampleDataset = [
            'the quick brown fox jumped over the lazy dog',
            'the quick dog runs fast'
        ];

        $this->model = new Word2Vec(100, 'neg', 2, 0, .05, 1000, 1);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function assertPreConditions() : void
    {
        $this->assertFalse($this->model->trained());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Word2Vec::class, $this->model);
    }

    /**
     * @test
     */
    public function badNumDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Word2Vec(0, 'neg', 2);
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'layer' => 'neg',
            'window' => 2,
            'dimensions' => 100,
            'sample_rate' => 0,
            'alpha' => .05,
            'epochs' => 1000,
            'min_count' => 1,
        ];

        $this->assertEquals($expected, $this->model->params());
    }

    /**
     * @test
     */
    public function trainPredict() : void
    {
        $this->model->train($this->sampleDataset);

        $this->assertTrue($this->model->trained());

        $mostSimilar = $this->model->mostSimilar(['dog']);
        $this->assertArrayHasKey('fast', $mostSimilar);

        $score = $mostSimilar['fast'];
        $this->assertGreaterThanOrEqual(.37, $score);
    }
}
