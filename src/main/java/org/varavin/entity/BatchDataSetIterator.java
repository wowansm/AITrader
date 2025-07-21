package org.varavin.entity;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.varavin.Config;

import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

public class BatchDataSetIterator implements DataSetIterator {

    private static final Logger log = LoggerFactory.getLogger(BatchDataSetIterator.class);

    private final INDArray features;
    private final INDArray labels;
    private final int batchSize;
    private int cursor = 0;
    private final int totalExamples;
    private final List<String> labelNames = Arrays.asList("UP", "DOWN", "SIDEWAYS");

    public BatchDataSetIterator(INDArray features, INDArray labels, int batchSize) {
        this.features = features;
        this.labels = labels;
        this.batchSize = batchSize;
        this.totalExamples = (int) features.size(0);
    }

    @Override
    public DataSet next(int num) {
        int actualBatchSize = Math.min(num, totalExamples - cursor);
        if (actualBatchSize <= 0) {
            throw new NoSuchElementException();
        }

        INDArray batchFeatures = features.get(
                NDArrayIndex.interval(cursor, cursor + actualBatchSize),
                NDArrayIndex.all(),
                NDArrayIndex.all()
        );

        INDArray batchLabels = labels.get(
                NDArrayIndex.interval(cursor, cursor + actualBatchSize),
                NDArrayIndex.all()
        );

        cursor += actualBatchSize;

        // --- ИЗМЕНЕНИЕ: Для CNN1D данные должны быть в формате [batch, features, timesteps] ---
        // Наш DataManager уже создает данные в формате [NUM_FEATURES, TIME_STEPS] для каждого примера,
        // поэтому при батчинге мы получаем [batch, NUM_FEATURES, TIME_STEPS], что является правильным форматом.
        // Никаких дополнительных перестановок (permute) не требуется.
        return new DataSet(batchFeatures, batchLabels);
    }

    @Override
    public int inputColumns() {
        // Для CNN это ширина (timesteps)
        return Config.TIME_STEPS;
    }

    @Override
    public int totalOutcomes() {
        return Config.NUM_OUTPUTS;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        // Not used
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return labelNames;
    }

    @Override
    public boolean hasNext() {
        return cursor < totalExamples;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
