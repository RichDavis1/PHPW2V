<?php

use PHPW2V\Word2Vec;
use PHPW2V\SoftmaxApproximators\NegativeSampling;
use PHPW2V\SoftmaxApproximators\HierarchicalSoftmax;
use PHPUnit\Framework\TestCase;

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

        $this->model = new Word2Vec(100, new NegativeSampling(), 2, 0, .05, 1000, 1);
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

        new Word2Vec(0, new NegativeSampling(), 2);
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'layer' => new NegativeSampling(),
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
    public function trainPredictNegativeSampling() : void
    {
        $this->model->train($this->sampleDataset);

        $this->assertTrue($this->model->trained());

        $mostSimilar = $this->model->mostSimilar(['dog']);
        $this->assertArrayHasKey('fast', $mostSimilar);

        $score = $mostSimilar['fast'];
        $this->assertGreaterThanOrEqual(.37, $score);
    }

    /**
     * @test
     */
    public function trainPredictHierarchicalSoftmax() : void
    {
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
            'a dog is the only thing on earth that loves you more than you love yourself',
        ];

        $samples = [];
        foreach ($sentences as $sentence) {
            $samples[] = [$sentence];
        }

        $model = new Word2Vec(150, new HierarchicalSoftmax(), 2, .05, .05, 350, 1);
        $model->train($sentences);

        $this->assertTrue($model->trained());

        $mostSimilar = $model->mostSimilar(['dog']);
        $this->assertArrayHasKey('fox', $mostSimilar);
        $this->assertArrayHasKey('pug', $mostSimilar);

        $this->assertGreaterThanOrEqual(.40, $mostSimilar['fox']);
        $this->assertGreaterThanOrEqual(.40, $mostSimilar['pug']);
    }
}
