package org.varavin;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ta4j.core.*;
import org.ta4j.core.indicators.*;
import org.ta4j.core.indicators.adx.ADXIndicator;
import org.ta4j.core.indicators.bollinger.BollingerBandsLowerIndicator;
import org.ta4j.core.indicators.bollinger.BollingerBandsMiddleIndicator;
import org.ta4j.core.indicators.bollinger.BollingerBandsUpperIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.helpers.VolumeIndicator;
import org.ta4j.core.indicators.statistics.StandardDeviationIndicator;
import org.ta4j.core.num.DoubleNum;
import org.ta4j.core.num.Num;
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

    public static BarSeries getOriginalSeries() {
        if (originalSeries == null) {
            loadOriginalSeries();
        }
        return originalSeries;
    }

    public static int getTestDataStartIndex() {
        return testDataStartIndex;
    }

    public static ProcessedData prepareData() {
        try {
            getOriginalSeries();
            int totalExamples = originalSeries.getBarCount() - Config.MAX_INDICATOR_PERIOD - Config.TIME_STEPS - Config.MAX_FUTURE_TICKS - 1;
            int trainEndOriginal = (int) (totalExamples * 0.7);
            int valEndOriginal = trainEndOriginal + (int) (totalExamples * 0.15);
            testDataStartIndex = valEndOriginal + Config.MAX_INDICATOR_PERIOD + Config.TIME_STEPS;
            log.info("Рассчитан консистентный testDataStartIndex: {}", testDataStartIndex);

            ProcessedData cachedData = loadProcessedDataFromCache();
            if (cachedData != null) {
                log.info("Используются кэшированные данные версии {}", Config.DATA_VERSION);
                return cachedData;
            }

            log.info("Кэш не найден. Начинаем подготовку данных с нуля.");
            Map<String, Indicator<Num>> indicators = calculateAllIndicators(originalSeries);

            List<INDArray> featuresList = new ArrayList<>();
            List<INDArray> labelsList = new ArrayList<>();

            for (int i = Config.MAX_INDICATOR_PERIOD + Config.TIME_STEPS; i < originalSeries.getBarCount() - Config.MAX_FUTURE_TICKS - 1; i++) {
                INDArray label = createClassificationLabel(originalSeries, indicators.get("ATR14"), i);
                INDArray featureWindow = createFeatureWindowAsINDArray(originalSeries, indicators, i);

                if (featureWindow != null && label != null) {
                    featuresList.add(featureWindow);
                    labelsList.add(label);
                }
            }

            if (featuresList.isEmpty()) throw new IllegalStateException("Не удалось создать признаки!");
            log.info("Создано примеров (до разделения): {}", featuresList.size());

            INDArray allFeatures = Nd4j.stack(0, featuresList.toArray(new INDArray[0]));
            INDArray allLabels = Nd4j.vstack(labelsList.toArray(new INDArray[0]));

            INDArray[] splitData = splitDataset(allFeatures, allLabels, trainEndOriginal, valEndOriginal);

            INDArray[] oversampledTrainData = oversample(splitData[0], splitData[1]);

            ProcessedData processedData = new ProcessedData(oversampledTrainData[0], oversampledTrainData[1], splitData[2], splitData[3], splitData[4], splitData[5]);
            saveProcessedDataToCache(processedData);
            return processedData;

        } catch (Exception e) {
            log.error("Ошибка подготовки данных: ", e);
            return null;
        }
    }

    private static INDArray[] oversample(INDArray features, INDArray labels) {
        log.info("Начало Oversampling...");
        INDArray classCounts = labels.sum(0);
        log.info("Количество примеров по классам (до): {}", classCounts);

        long majorityCount = classCounts.maxNumber().longValue();
        int minorityClassIndex = 2; // SIDEWAYS

        List<INDArray> newFeatures = new ArrayList<>();
        List<INDArray> newLabels = new ArrayList<>();

        for (int i = 0; i < features.size(0); i++) {
            newFeatures.add(features.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()));
            newLabels.add(labels.getRow(i));
        }

        List<Integer> minorityIndices = new ArrayList<>();
        for (int i = 0; i < labels.size(0); i++) {
            if (labels.getDouble(i, minorityClassIndex) == 1.0) {
                minorityIndices.add(i);
            }
        }

        if (!minorityIndices.isEmpty()) {
            long currentMinorityCount = minorityIndices.size();
            long samplesToAdd = majorityCount - currentMinorityCount;
            Random rand = new Random();
            for (int i = 0; i < samplesToAdd; i++) {
                int randomIndex = minorityIndices.get(rand.nextInt(minorityIndices.size()));
                newFeatures.add(features.get(NDArrayIndex.point(randomIndex), NDArrayIndex.all(), NDArrayIndex.all()));
                newLabels.add(labels.getRow(randomIndex));
            }
        }

        INDArray finalFeatures = Nd4j.stack(0, newFeatures.toArray(new INDArray[0]));
        INDArray finalLabels = Nd4j.vstack(newLabels.toArray(new INDArray[0]));

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < finalFeatures.size(0); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random());

        INDArray shuffledFeatures = Nd4j.zeros(finalFeatures.shape());
        INDArray shuffledLabels = Nd4j.zeros(finalLabels.shape());

        for (int i = 0; i < indices.size(); i++) {
            int originalIndex = indices.get(i);
            INDArray featureSlice = finalFeatures.get(NDArrayIndex.point(originalIndex), NDArrayIndex.all(), NDArrayIndex.all());
            shuffledFeatures.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()}, featureSlice);
            shuffledLabels.putRow(i, finalLabels.getRow(originalIndex));
        }

        log.info("Количество примеров по классам (после): {}", shuffledLabels.sum(0));
        log.info("Oversampling завершен. Размер тренировочного набора: {}", shuffledFeatures.size(0));

        return new INDArray[]{shuffledFeatures, shuffledLabels};
    }

    private static INDArray[] splitDataset(INDArray features, INDArray labels, int trainEnd, int valEnd) {
        int total = (int) features.size(0);
        return new INDArray[]{features.get(NDArrayIndex.interval(0, trainEnd), NDArrayIndex.all(), NDArrayIndex.all()), labels.get(NDArrayIndex.interval(0, trainEnd), NDArrayIndex.all()), features.get(NDArrayIndex.interval(trainEnd, valEnd), NDArrayIndex.all(), NDArrayIndex.all()), labels.get(NDArrayIndex.interval(trainEnd, valEnd), NDArrayIndex.all()), features.get(NDArrayIndex.interval(valEnd, total), NDArrayIndex.all(), NDArrayIndex.all()), labels.get(NDArrayIndex.interval(valEnd, total), NDArrayIndex.all())};
    }

    private static INDArray createClassificationLabel(BarSeries series, Indicator<Num> atr, int currentIndex) {
        int futureEndIndex = currentIndex + Config.MAX_FUTURE_TICKS;
        if (futureEndIndex >= series.getBarCount()) return null;
        double currentPrice = series.getBar(currentIndex).getClosePrice().doubleValue();
        double currentAtr = atr.getValue(currentIndex).doubleValue();
        if (currentAtr < 1e-6) return null;
        double upTargetPrice = currentPrice + Config.CLASSIFICATION_ATR_UP_THRESHOLD * currentAtr;
        double downTargetPrice = currentPrice - Config.CLASSIFICATION_ATR_DOWN_THRESHOLD * currentAtr;
        for (int j = currentIndex + 1; j <= futureEndIndex; j++) {
            Bar futureBar = series.getBar(j);
            if (futureBar.getLowPrice().doubleValue() <= downTargetPrice) return Nd4j.create(new float[]{0, 1, 0});
            if (futureBar.getHighPrice().doubleValue() >= upTargetPrice) return Nd4j.create(new float[]{1, 0, 0});
        }
        return Nd4j.create(new float[]{0, 0, 1});
    }

    private static INDArray createFeatureWindowAsINDArray(BarSeries series, Map<String, Indicator<Num>> indicators, int currentIndex) {
        try {
            INDArray window = Nd4j.zeros(Config.NUM_FEATURES, Config.TIME_STEPS);

            for (int j = 0; j < Config.TIME_STEPS; j++) {
                int idx = currentIndex - Config.TIME_STEPS + 1 + j;
                if (idx <= 0) return null;

                Bar bar = series.getBar(idx);
                int featureIdx = 0;

                double close = bar.getClosePrice().doubleValue();
                window.putScalar(featureIdx++, j, bar.getOpenPrice().doubleValue() / close - 1);
                window.putScalar(featureIdx++, j, bar.getHighPrice().doubleValue() / close - 1);
                window.putScalar(featureIdx++, j, bar.getLowPrice().doubleValue() / close - 1);
                window.putScalar(featureIdx++, j, 0.0);

                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("Volume"), idx));
                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("RSI14"), idx));
                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("ATR14"), idx));

                ZonedDateTime endTime = bar.getEndTime();
                window.putScalar(featureIdx++, j, endTime.getDayOfWeek().getValue());
                window.putScalar(featureIdx++, j, endTime.getHour());

                double macd = getIndicatorValueSafe(indicators.get("MACD"), idx);
                double macdSignal = getIndicatorValueSafe(indicators.get("MACD_Signal"), idx);
                window.putScalar(featureIdx++, j, macd - macdSignal);

                double bbMiddle = getIndicatorValueSafe(indicators.get("BB_Middle"), idx);
                double bbUpper = getIndicatorValueSafe(indicators.get("BB_Upper"), idx);
                double bbLower = getIndicatorValueSafe(indicators.get("BB_Lower"), idx);
                window.putScalar(featureIdx++, j, (bbUpper - bbLower));
                window.putScalar(featureIdx++, j, (close - bbLower) / (bbUpper - bbLower));

                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("StochK"), idx));
                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("StochD"), idx));

                double sma200 = getIndicatorValueSafe(indicators.get("SMA200"), idx);
                window.putScalar(featureIdx++, j, close / sma200 - 1);

                double atr100 = getIndicatorValueSafe(indicators.get("ATR100"), idx);
                double atr14 = getIndicatorValueSafe(indicators.get("ATR14"), idx);
                window.putScalar(featureIdx++, j, atr14 / atr100);

                // --- НОВЫЙ ПРИЗНАК ---
                window.putScalar(featureIdx++, j, getIndicatorValueSafe(indicators.get("ADX14"), idx));
            }

            for (int f = 0; f < Config.NUM_FEATURES; f++) {
                INDArray featureRow = window.getRow(f);
                double firstValue = featureRow.getDouble(0);
                if (Math.abs(firstValue) > 1e-7) {
                    featureRow.divi(firstValue).subi(1.0);
                }
            }
            return window;
        } catch (Exception e) {
            log.warn("Ошибка создания окна признаков на индексе {}: {}", currentIndex, e.getMessage());
            return null;
        }
    }

    private static Map<String, Indicator<Num>> calculateAllIndicators(BarSeries series) {
        Map<String, Indicator<Num>> indicators = new LinkedHashMap<>();
        ClosePriceIndicator closePrice = new ClosePriceIndicator(series);
        try {
            indicators.put("Volume", new VolumeIndicator(series));
            indicators.put("RSI14", new RSIIndicator(closePrice, 14));
            indicators.put("ATR14", new ATRIndicator(series, 14));

            MACDIndicator macd = new MACDIndicator(closePrice, 12, 26);
            indicators.put("MACD", macd);
            indicators.put("MACD_Signal", new EMAIndicator(macd, 9));

            BollingerBandsMiddleIndicator bbMiddle = new BollingerBandsMiddleIndicator(closePrice);
            StandardDeviationIndicator sd = new StandardDeviationIndicator(closePrice, 20);
            indicators.put("BB_Middle", bbMiddle);
            indicators.put("BB_Upper", new BollingerBandsUpperIndicator(bbMiddle, sd));
            indicators.put("BB_Lower", new BollingerBandsLowerIndicator(bbMiddle, sd));

            StochasticOscillatorKIndicator stochK = new StochasticOscillatorKIndicator(series, 14);
            indicators.put("StochK", stochK);
            indicators.put("StochD", new SMAIndicator(stochK, 3));

            indicators.put("SMA200", new SMAIndicator(closePrice, 200));
            indicators.put("ATR100", new ATRIndicator(series, 100));

            // --- НОВЫЙ ИНДИКАТОР ---
            indicators.put("ADX14", new ADXIndicator(series, 14));

        } catch (Exception e) {
            log.error("Ошибка инициализации индикаторов: ", e);
        }
        return indicators;
    }

    private static void loadOriginalSeries() {
        if (originalSeries == null) {
            try {
                RecordReader recordReader = new CSVRecordReader(0, ',');
                recordReader.initialize(new FileSplit(new File(Config.CSV_FILE_NAME)));
                originalSeries = loadSeriesFromReader(recordReader);
                log.info("Загружена оригинальная серия из {} баров", originalSeries.getBarCount());
            } catch (Exception e) {
                log.error("Не удалось загрузить оригинальную серию для бэктестинга", e);
            }
        }
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
                Bar bar = new BaseBar(Duration.ofMinutes(5), time, Double.parseDouble(record.get(1).toString()), Double.parseDouble(record.get(2).toString()), Double.parseDouble(record.get(3).toString()), Double.parseDouble(record.get(4).toString()), Double.parseDouble(record.get(5).toString()));
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

    private static void saveProcessedDataToCache(ProcessedData data) throws IOException {
        File cacheDir = new File(Config.CACHE_DIR);
        if (!cacheDir.exists() && !cacheDir.mkdirs()) {
            throw new IOException("Не удалось создать директорию кэша");
        }
        String prefix = Config.DATA_VERSION + "_";
        saveINDArray(new File(cacheDir, prefix + "trainFeatures.bin"), data.trainFeatures());
        saveINDArray(new File(cacheDir, prefix + "trainLabels.bin"), data.trainLabels());
        saveINDArray(new File(cacheDir, prefix + "valFeatures.bin"), data.valFeatures());
        saveINDArray(new File(cacheDir, prefix + "valLabels.bin"), data.valLabels());
        saveINDArray(new File(cacheDir, prefix + "testFeatures.bin"), data.testFeatures());
        saveINDArray(new File(cacheDir, prefix + "testLabels.bin"), data.testLabels());
    }

    private static void saveINDArray(File file, INDArray array) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            Nd4j.write(array, dos);
        }
    }

    private static ProcessedData loadProcessedDataFromCache() {
        try {
            File cacheDir = new File(Config.CACHE_DIR);
            if (!cacheDir.exists()) return null;
            String prefix = Config.DATA_VERSION + "_";
            File trainFeaturesFile = new File(cacheDir, prefix + "trainFeatures.bin");
            if (!trainFeaturesFile.exists()) return null;

            return new ProcessedData(loadINDArray(trainFeaturesFile), loadINDArray(new File(cacheDir, prefix + "trainLabels.bin")), loadINDArray(new File(cacheDir, prefix + "valFeatures.bin")), loadINDArray(new File(cacheDir, prefix + "valLabels.bin")), loadINDArray(new File(cacheDir, prefix + "testFeatures.bin")), loadINDArray(new File(cacheDir, prefix + "testLabels.bin")));
        } catch (Exception e) {
            log.warn("Не удалось загрузить кэшированные данные: {}", e.getMessage());
            return null;
        }
    }

    private static INDArray loadINDArray(File file) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            return Nd4j.read(dis);
        }
    }
}
