import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.NoSuchElementException;

/**
 * Created by tigershark on 11/25/15.
 */
public class EpisodeIterator implements DataSetIterator {

    /**
     * Number of episodes to return on each next().
     */
    private int miniBatchSize = 1;

    /**
     * Pointer to the internal array of data.
     */
    private int examplesSoFar = 0;

    /**
     * Total number of episodes in this data set. Must be a multiple of miniBatchSize.
     */
    private int numExamplesToFetch = 1;

    private float[] stateData;
    private float[] rewardData;


    public EpisodeIterator(float[] stateData, float[] rewardData, int batchSize) {
        this.stateData = stateData;
        this.rewardData = rewardData;
        this.miniBatchSize = batchSize;
    }

    @Override
    public int totalExamples() {
        return numExamplesToFetch;
    }

    @Override
    public int inputColumns() {
        return this.stateData.length;
    }

    @Override
    public int totalOutcomes() {
        return this.rewardData.length;
    }

    @Override
    public void reset() {
        this.examplesSoFar = 0;
    }

    @Override
    public int batch() {
        return this.miniBatchSize;
    }

    @Override
    public int cursor() {
        return this.examplesSoFar;
    }

    @Override
    public int numExamples() {
        return this.numExamplesToFetch;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public boolean hasNext() {
        return examplesSoFar + miniBatchSize <= numExamplesToFetch;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if (examplesSoFar + num > numExamplesToFetch) throw new NoSuchElementException();
        //Allocate space:
        INDArray input = Nd4j.create(Arrays.copyOfRange(this.stateData, this.examplesSoFar, this.examplesSoFar + num));
        INDArray labels = Nd4j.create(Arrays.copyOfRange(this.rewardData, this.examplesSoFar, this.examplesSoFar + num));

        examplesSoFar += num;
        return new DataSet(input, labels);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not implemented");
    }
}
