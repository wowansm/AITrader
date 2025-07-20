package org.varavin;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ta4j.core.*;
import org.ta4j.core.indicators.*;
import org.ta4j.core.indicators.helpers.*;
import org.ta4j.core.num.DoubleNum;
import org.ta4j.core.num.Num;
import org.varavin.entity.BatchDataSetIterator;
import org.varavin.entity.ProcessedData;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.*;

public class DataManager {
    private static final Logger log = LoggerFactory.getLogger(DataManager.class);

    private static BarSeries originalSeries = null;
    private static int testDataStartIndex = -1;
    private static Indicator<Num> atrIndicator = null;

    public static BarSeries getOriginalSeries() {
        return originalSeries;
    }

    public static int getTestDataStartIndex() {
        return testDataStartIndex;
    }


    public static DataSetIterator[] prepareData(int batchSize) {
        try {
            ProcessedData cachedData = loadProcessedData();
            if (cachedData != null) {
                log.info("Используются кэшированные данные версии {}", Config.DATA_VERSION);
                loadOriginalSeries();
                Map<String, Indicator<Num>> indicators = calculateSimpleIndicators(originalSeries);
                atrIndicator = indicators.get("ATR14");
                long trainSize = cachedData.trainFeatures.size(0);
                long valSize = cachedData.valFeatures.size(0);
                testDataStartIndex = (int)(trainSize + valSize) + Config.MAX_INDICATOR_PERIOD + Config.TIME_STEPS;
                log.info("Восстановлен testDataStartIndex из кэша: {}", testDataStartIndex);
                return createIteratorsFromProcessedData(cachedData.trainFeatures, cachedData.trainLabels, cachedData.valFeatures, cachedData.valLabels, cachedData.testFeatures, cachedData.testLabels, batchSize);
            }
            log.info("Кэш не найден. Начинаем подготовку данных с нуля.");

            loadOriginalSeries();
            Map<String, Indicator<Num>> indicators = calculateSimpleIndicators(originalSeries);
            atrIndicator = indicators.get("ATR14");

            List<INDArray> featuresList = new ArrayList<>();
            List<INDArray> labelsList = new ArrayList<>();

            for (int i = Config.MAX_INDICATOR_PERIOD + Config.TIME_STEPS; i < originalSeries.getBarCount() - Config.MAX_FUTURE_TICKS - 1; i++) {
                INDArray label = createVolatilityNormalizedLabel(originalSeries, atrIndicator, i);
                INDArray featureWindow = createFeatureWindowAsINDArray(originalSeries, indicators, i);

                if (featureWindow != null && label != null) {
                    featuresList.add(featureWindow);
                    labelsList.add(label);
                }
            }

            if (featuresList.isEmpty()) throw new IllegalStateException("Не удалось создать признаки!");
            log.info("Создано примеров: {}", featuresList.size());

            INDArray allFeatures = Nd4j.stack(0, featuresList.toArray(new INDArray[0]));
            INDArray allLabels = Nd4j.vstack(labelsList.toArray(new INDArray[0]));

            INDArray[] splitData = splitDataset(allFeatures, allLabels, 0.7, 0.15);
            INDArray[] normalizedFeatures = normalizeFeaturesZScore(splitData[0], splitData[2], splitData[4]);

            saveProcessedData(
                    normalizedFeatures[0], splitData[1],
                    normalizedFeatures[1], splitData[3],
                    normalizedFeatures[2], splitData[5]
            );

            return createIteratorsFromProcessedData(
                    normalizedFeatures[0], splitData[1],
                    normalizedFeatures[1], splitData[3],
                    normalizedFeatures[2], splitData[5],
                    batchSize
            );

        } catch (Exception e) {
            log.error("Ошибка подготовки данных: ", e);
            return null;
        }
    }

    private static INDArray createVolatilityNormalizedLabel(BarSeries series, Indicator<Num> atr, int currentIndex) {
        int futureEndIndex = currentIndex + Config.MAX_FUTURE_TICKS;
        if (futureEndIndex >= series.getBarCount()) return null;

        double currentPrice = series.getBar(currentIndex).getClosePrice().doubleValue();
        double currentAtr = atr.getValue(currentIndex).doubleValue();
        if (currentAtr < 1e-6) return null;

        double maxFuturePrice = currentPrice;
        double minFuturePrice = currentPrice;

        for (int j = currentIndex + 1; j <= futureEndIndex; j++) {
            Bar futureBar = series.getBar(j);
            if (futureBar.getHighPrice().doubleValue() > maxFuturePrice) maxFuturePrice = futureBar.getHighPrice().doubleValue();
            if (futureBar.getLowPrice().doubleValue() < minFuturePrice) minFuturePrice = futureBar.getLowPrice().doubleValue();
        }

        double potentialGainAtr = (maxFuturePrice - currentPrice) / currentAtr;
        double potentialLossAtr = (currentPrice - minFuturePrice) / currentAtr;

        return Nd4j.create(new double[]{potentialGainAtr, potentialLossAtr});
    }

    private static double calculatePctChange(double current, double previous) {
        if (previous == 0 || Math.abs(previous) < 1e-9) return 0.0;
        return (current - previous) / previous;
    }

    // --- ИЗМЕНЕНИЕ: Новый метод для создания упрощенного набора признаков ---
    private static INDArray createFeatureWindowAsINDArray(BarSeries series, Map<String, Indicator<Num>> indicators, int currentIndex) {
        try {
            INDArray window = Nd4j.zeros(Config.NUM_FEATURES, Config.TIME_STEPS);

            for (int j = 0; j < Config.TIME_STEPS; j++) {
                int idx = currentIndex - Config.TIME_STEPS + 1 + j;
                if (idx <= 0) return null;

                Bar bar = series.getBar(idx);
                Bar prevBar = series.getBar(idx - 1);

                int featureIdx = 0;
                // 1. Price Action (4 признака)
                window.putScalar(featureIdx++, j, calculatePctChange(bar.getOpenPrice().doubleValue(), prevBar.getClosePrice().doubleValue()));
                window.putScalar(featureIdx++, j, calculatePctChange(bar.getHighPrice().doubleValue(), bar.getOpenPrice().doubleValue()));
                window.putScalar(featureIdx++, j, calculatePctChange(bar.getLowPrice().doubleValue(), bar.getOpenPrice().doubleValue()));
                window.putScalar(featureIdx++, j, calculatePctChange(bar.getClosePrice().doubleValue(), bar.getOpenPrice().doubleValue()));

                // 2. Volume (1 признак)
                window.putScalar(featureIdx++, j, calculatePctChange(bar.getVolume().doubleValue(), prevBar.getVolume().doubleValue()));

                // 3. Momentum (1 признак)
                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("RSI14"), idx));

                // 4. Volatility (1 признак)
                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("ATR14"), idx));

                // 5. Time (2 признака)
                ZonedDateTime endTime = bar.getEndTime();
                window.putScalar(featureIdx++, j, (double) endTime.getDayOfWeek().getValue() / 7.0);
                window.putScalar(featureIdx++, j, (double) endTime.getHour() / 23.0);
            }
            return window;
        } catch (Exception e) {
            log.warn("Ошибка создания окна признаков на индексе {}: {}", currentIndex, e.getMessage());
            return null;
        }
    }

    // --- ИЗМЕНЕНИЕ: Считаем только необходимые индикаторы ---
    private static Map<String, Indicator<Num>> calculateSimpleIndicators(BarSeries series) {
        Map<String, Indicator<Num>> indicators = new LinkedHashMap<>();
        ClosePriceIndicator closePrice = new ClosePriceIndicator(series);
        try {
            indicators.put("RSI14", new RSIIndicator(closePrice, 14));
            indicators.put("ATR14", new ATRIndicator(series, 14));
        } catch (Exception e) {
            log.error("Ошибка инициализации индикаторов: ", e);
        }
        return indicators;
    }

    // --- Остальной код без изменений ---
    public static double getOriginalPrice(int testStep) {
        if (originalSeries != null && testDataStartIndex != -1) {
            int originalIndex = testDataStartIndex + testStep;
            if (originalIndex < originalSeries.getBarCount()) {
                return originalSeries.getBar(originalIndex).getClosePrice().doubleValue();
            }
        }
        return -1;
    }

    private static void loadOriginalSeries() {
        if (originalSeries == null) {
            try {
                RecordReader recordReader = new CSVRecordReader(0, ',');
                recordReader.initialize(new FileSplit(new File(Config.CSV_FILE_NAME)));
                originalSeries = loadSeriesFromReader(recordReader);
            } catch (Exception e) {
                log.error("Failed to load original series for backtesting", e);
            }
        }
    }

    private static INDArray[] splitDataset(INDArray features, INDArray labels, double trainRatio, double valRatio) {
        int total = (int) features.size(0);
        int trainEnd = (int) (total * trainRatio);
        int valEnd = trainEnd + (int) (total * valRatio);
        testDataStartIndex = valEnd + Config.MAX_INDICATOR_PERIOD + Config.TIME_STEPS;
        log.info("Индекс начала тестовых данных: {}", testDataStartIndex);

        return new INDArray[]{
                features.get(NDArrayIndex.interval(0, trainEnd), NDArrayIndex.all(), NDArrayIndex.all()),
                labels.get(NDArrayIndex.interval(0, trainEnd), NDArrayIndex.all()),
                features.get(NDArrayIndex.interval(trainEnd, valEnd), NDArrayIndex.all(), NDArrayIndex.all()),
                labels.get(NDArrayIndex.interval(trainEnd, valEnd), NDArrayIndex.all()),
                features.get(NDArrayIndex.interval(valEnd, total), NDArrayIndex.all(), NDArrayIndex.all()),
                labels.get(NDArrayIndex.interval(valEnd, total), NDArrayIndex.all())
        };
    }

    private static INDArray[] normalizeFeaturesZScore(INDArray trainFeatures, INDArray valFeatures, INDArray testFeatures) {
        INDArray featureStats = Nd4j.zeros(Config.NUM_FEATURES, 2);
        for (int f = 0; f < Config.NUM_FEATURES; f++) {
            INDArray trainFeatureSlice = trainFeatures.get(NDArrayIndex.all(), NDArrayIndex.point(f), NDArrayIndex.all());
            double mean = trainFeatureSlice.meanNumber().doubleValue();
            double stdDev = trainFeatureSlice.stdNumber().doubleValue();
            if (stdDev < 1e-8) stdDev = 1.0;
            featureStats.putScalar(f, 0, mean);
            featureStats.putScalar(f, 1, stdDev);
        }
        saveFeatureStats(featureStats);
        return new INDArray[]{
                normalizeFeatureSetZScore(trainFeatures, featureStats),
                normalizeFeatureSetZScore(valFeatures, featureStats),
                normalizeFeatureSetZScore(testFeatures, featureStats)
        };
    }

    private static INDArray normalizeFeatureSetZScore(INDArray features, INDArray featureStats) {
        INDArray normalized = features.dup();
        for (int f = 0; f < Config.NUM_FEATURES; f++) {
            double mean = featureStats.getDouble(f, 0);
            double stdDev = featureStats.getDouble(f, 1);
            if (stdDev > 1e-8) {
                INDArray featureSlice = normalized.get(NDArrayIndex.all(), NDArrayIndex.point(f), NDArrayIndex.all());
                featureSlice.subi(mean).divi(stdDev);
            }
        }
        return normalized;
    }

    private static double getIndicatorValueSafe(Indicator<Num> indicator, int index) {
        try {
            if (index >= 0 && index < indicator.getBarSeries().getBarCount()) {
                Num value = indicator.getValue(index);
                if (value != null) {
                    double doubleValue = value.doubleValue();
                    return Double.isFinite(doubleValue) ? doubleValue : 0.0;
                }
            }
            return 0.0;
        } catch (Exception e) {
            return 0.0;
        }
    }

    private static BarSeries loadSeriesFromReader(RecordReader reader) throws IOException, InterruptedException {
        BarSeries series = new BaseBarSeriesBuilder().withNumTypeOf(DoubleNum.class).withName("GAZPROM").build();
        List<Bar> bars = new ArrayList<>();
        while (reader.hasNext()) {
            List<Writable> record = reader.next();
            try {
                ZonedDateTime time = Instant.ofEpochSecond(Long.parseLong(record.get(0).toString())).atZone(ZoneId.systemDefault());
                Bar bar = new BaseBar(Duration.ofMinutes(5), time,
                        Double.parseDouble(record.get(1).toString()),
                        Double.parseDouble(record.get(2).toString()),
                        Double.parseDouble(record.get(3).toString()),
                        Double.parseDouble(record.get(4).toString()),
                        Double.parseDouble(record.get(5).toString()));
                bars.add(bar);
            } catch (Exception e) {
                log.warn("Пропуск некорректной записи: {}", e.getMessage());
            }
        }
        bars.sort(Comparator.comparing(Bar::getEndTime));
        for (Bar bar : bars) {
            series.addBar(bar);
        }
        return series;
    }

    private static DataSetIterator[] createIteratorsFromProcessedData(
            INDArray trainFeatures, INDArray trainLabels,
            INDArray valFeatures, INDArray valLabels,
            INDArray testFeatures, INDArray testLabels,
            int batchSize) {
        log.info("Создание итераторов для данных:");
        log.info("Train: {} примеров", trainFeatures.size(0));
        log.info("Val: {} примеров", valFeatures.size(0));
        log.info("Test: {} примеров", testFeatures.size(0));
        return new DataSetIterator[]{
                new BatchDataSetIterator(trainFeatures, trainLabels, batchSize),
                new BatchDataSetIterator(valFeatures, valLabels, batchSize),
                new BatchDataSetIterator(testFeatures, testLabels, batchSize)
        };
    }
    private static void saveFeatureStats(INDArray featureStats) {
        try {
            File cacheDir = new File(Config.CACHE_DIR);
            if (!cacheDir.exists()) cacheDir.mkdirs();
            File statsFile = new File(cacheDir, Config.DATA_VERSION + "_feature_stats.bin");
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(statsFile))) {
                Nd4j.write(featureStats, dos);
            }
        } catch (IOException e) {
            log.warn("Не удалось сохранить статистику признаков: {}", e.getMessage());
        }
    }
    private static void saveProcessedData(INDArray trainFeatures, INDArray trainLabels,
                                          INDArray valFeatures, INDArray valLabels,
                                          INDArray testFeatures, INDArray testLabels) throws IOException {
        File cacheDir = new File(Config.CACHE_DIR);
        if (!cacheDir.exists() && !cacheDir.mkdirs()) {
            throw new IOException("Не удалось создать директорию кэша");
        }
        String prefix = Config.DATA_VERSION + "_";
        saveINDArray(new File(cacheDir, prefix + "trainFeatures.bin"), trainFeatures);
        saveINDArray(new File(cacheDir, prefix + "trainLabels.bin"), trainLabels);
        saveINDArray(new File(cacheDir, prefix + "valFeatures.bin"), valFeatures);
        saveINDArray(new File(cacheDir, prefix + "valLabels.bin"), valLabels);
        saveINDArray(new File(cacheDir, prefix + "testFeatures.bin"), testFeatures);
        saveINDArray(new File(cacheDir, prefix + "testLabels.bin"), testLabels);
    }
    private static void saveINDArray(File file, INDArray array) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            Nd4j.write(array, dos);
        }
    }
    private static ProcessedData loadProcessedData() {
        try {
            File cacheDir = new File(Config.CACHE_DIR);
            if (!cacheDir.exists()) return null;
            String prefix = Config.DATA_VERSION + "_";
            File trainFeaturesFile = new File(cacheDir, prefix + "trainFeatures.bin");
            if (!trainFeaturesFile.exists()) return null;
            INDArray trainFeatures = loadINDArray(trainFeaturesFile);
            INDArray trainLabels = loadINDArray(new File(cacheDir, prefix + "trainLabels.bin"));
            INDArray valFeatures = loadINDArray(new File(cacheDir, prefix + "valFeatures.bin"));
            INDArray valLabels = loadINDArray(new File(cacheDir, prefix + "valLabels.bin"));
            INDArray testFeatures = loadINDArray(new File(cacheDir, prefix + "testFeatures.bin"));
            INDArray testLabels = loadINDArray(new File(cacheDir, prefix + "testLabels.bin"));
            return new ProcessedData(trainFeatures, trainLabels, valFeatures, valLabels, testFeatures, testLabels);
        } catch (Exception e) {
            log.warn("Не удалось загрузить кэшированные данные: {}", e.getMessage());
            return null;
        }
    }

    static INDArray loadINDArray(File file) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            return Nd4j.read(dis);
        }
    }

    public static double getOriginalAtr(int testStep) {
        if (atrIndicator != null && testDataStartIndex != -1) {
            int originalIndex = testDataStartIndex + testStep;
            if (originalIndex < atrIndicator.getBarSeries().getBarCount()) {
                double atrValue = atrIndicator.getValue(originalIndex).doubleValue();
                return Double.isFinite(atrValue) ? atrValue : -1.0;
            }
        }
        return -1.0;
    }
}
