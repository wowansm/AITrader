package org.varavin.entity;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.varavin.Config;
import org.varavin.DataManager;

import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

public class BatchDataSetIterator implements DataSetIterator {

    private static final Logger log = LoggerFactory.getLogger(BatchDataSetIterator.class);

    private final INDArray features;
    private final INDArray labels;
    private final int batchSize;
    private int cursor = 0;
    private final int totalExamples;

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

        // DL4J for RNN expects format [minibatch, features, timesteps]
        // We prepare data as [minibatch, timesteps, features], so we permute here.
        return new DataSet(batchFeatures, batchLabels);
    }

    @Override
    public int inputColumns() {
        return Config.NUM_FEATURES;
    }

    @Override
    public int totalOutcomes() {
        // For regression, this is the number of output variables
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
        // Not applicable for regression
        return Collections.emptyList();
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
