package org.varavin.entity;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ProcessedData {
    public final INDArray trainFeatures, trainLabels;
    public final INDArray valFeatures, valLabels;
    public final INDArray testFeatures, testLabels;

    public ProcessedData(INDArray trainFeatures, INDArray trainLabels,
                         INDArray valFeatures, INDArray valLabels,
                         INDArray testFeatures, INDArray testLabels) {
        this.trainFeatures = trainFeatures;
        this.trainLabels = trainLabels;
        this.valFeatures = valFeatures;
        this.valLabels = valLabels;
        this.testFeatures = testFeatures;
        this.testLabels = testLabels;
    }
}