package org.varavin.entity;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * ИСПРАВЛЕНИЕ: Класс преобразован в record.
 * Это автоматически создает канонический конструктор и публичные
 * методы доступа (например, testFeatures(), testLabels() и т.д.),
 * что соответствует ожиданиям вызывающего кода в ParameterOptimizer.
 */
public record ProcessedData(
        INDArray trainFeatures, INDArray trainLabels,
        INDArray valFeatures, INDArray valLabels,
        INDArray testFeatures, INDArray testLabels
) {
}
