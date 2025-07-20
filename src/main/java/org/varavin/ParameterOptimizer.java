package org.varavin;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.varavin.entity.BotParameters;

import java.io.File;
import java.io.IOException;

public class ParameterOptimizer {

    private static final Logger log = LoggerFactory.getLogger(ParameterOptimizer.class);

    public static void main(String[] args) throws IOException {
        log.info("--- Подготовка данных для оптимизации ---");
        DataSetIterator[] data = DataManager.prepareData(Config.BATCH_SIZE);
        if (data == null) {
            log.error("Не удалось подготовить данные. Оптимизация прервана.");
            return;
        }
        DataSetIterator testIter = data[2];

        File bestModelFile = new File(Config.MODEL_DIR, "bestModel.bin");
        if (!bestModelFile.exists()) {
            log.error("Файл лучшей модели не найден: {}. Оптимизация невозможна.", bestModelFile.getAbsolutePath());
            return;
        }
        MultiLayerNetwork bestModel = ModelSerializer.restoreMultiLayerNetwork(bestModelFile);
        log.info("Загружена модель для оптимизации: {}", bestModelFile.getName());

        // --- Новые диапазоны для новой сигнальной логики ---
        double atrStopStart = 1.5;
        double atrStopEnd = 5.0;
        double atrStopStep = 0.5;

        double rrStart = 1.0;
        double rrEnd = 3.0;
        double rrStep = 0.25;

        double thresholdStart = 0.3;
        double thresholdEnd = 1.5;
        double thresholdStep = 0.2;

        double riskPercentStart = 0.02; // 2%
        double riskPercentEnd = 0.10;   // 10%
        double riskPercentStep = 0.02;  // Шаг 2%

        optimize(bestModel, testIter,
                atrStopStart, atrStopEnd, atrStopStep,
                rrStart, rrEnd, rrStep,
                thresholdStart, thresholdEnd, thresholdStep,
                riskPercentStart, riskPercentEnd, riskPercentStep);
    }

    private static void optimize(MultiLayerNetwork model, DataSetIterator testIterator,
                                 double atrStopStart, double atrStopEnd, double atrStopStep,
                                 double rrStart, double rrEnd, double rrStep,
                                 double thresholdStart, double thresholdEnd, double thresholdStep,
                                 double riskStart, double riskEnd, double riskStep) {

        log.info("\n--- НАЧАЛО ОПТИМИЗАЦИИ ПАРАМЕТРОВ (СИГНАЛЬНАЯ ЛОГИКА) ---");
        log.info("Диапазон ATR Stop Multiplier: [{}...{}]", atrStopStart, atrStopEnd);
        log.info("Диапазон Fixed R/R Ratio:   [{}...{}]", rrStart, rrEnd);
        log.info("Диапазон Signal Threshold:  [{}...{}]", thresholdStart, thresholdEnd);
        log.info("Диапазон Risk Percent:      [{}...{}]", riskStart, riskEnd);
        log.info("----------------------------------------------------------");

        BotParameters bestParams = null;
        double bestBalance = -Double.MAX_VALUE;
        int totalIterations = 0;

        for (double risk = riskStart; risk <= riskEnd; risk += riskStep) {
            for (double atrStop = atrStopStart; atrStop <= atrStopEnd; atrStop += atrStopStep) {
                for (double rr = rrStart; rr <= rrEnd; rr += rrStep) {
                    for (double threshold = thresholdStart; threshold <= thresholdEnd; threshold += thresholdStep) {

                        totalIterations++;

                        BotParameters currentParams = new BotParameters(atrStop, rr, threshold, risk);

                        TradingBot bot = new TradingBot(currentParams, false);
                        testIterator.reset();
                        TradingBot.SimulationResult result = bot.runSimulation(model, testIterator);

                        if (result == null || result.totalTrades() < 10) { // Ищем более активные стратегии
                            continue;
                        }

                        if (result.finalBalance() > bestBalance) {
                            bestBalance = result.finalBalance();
                            bestParams = currentParams;
                            log.info(String.format("НОВЫЙ ЛИДЕР: Баланс: %.2f | Сделок: %d | ПФ: %.2f | Risk: %.0f%%, ATR Stop: %.1f, R/R: %.2f, Thresh: %.1f",
                                    bestBalance, result.totalTrades(), result.profitFactor(),
                                    bestParams.riskPercent() * 100, bestParams.atrStopMultiplier(),
                                    bestParams.fixedRiskRewardRatio(), bestParams.signalThreshold()));
                        }
                    }
                }
            }
        }

        log.info("\n--- ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ---");
        log.info("Всего итераций: {}", totalIterations);
        if (bestParams != null) {
            log.info("Лучший результат: Баланс = {}", String.format("%.2f", bestBalance));
            log.info("Лучшие параметры:");
            log.info("\tRISK_PER_TRADE_PERCENT = {}", bestParams.riskPercent());
            log.info("\tATR_STOP_MULTIPLIER = {}", bestParams.atrStopMultiplier());
            log.info("\tFIXED_RISK_REWARD_RATIO = {}", bestParams.fixedRiskRewardRatio());
            log.info("\tSIGNAL_THRESHOLD = {}", bestParams.signalThreshold());
        } else {
            log.warn("Не удалось найти оптимальные параметры. Попробуйте расширить диапазоны поиска или проверить логику модели.");
        }
    }
}
