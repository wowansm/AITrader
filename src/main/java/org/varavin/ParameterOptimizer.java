package org.varavin;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.varavin.entity.BotParameters;
import org.varavin.entity.ProcessedData;

import java.io.File;
import java.io.IOException;

public class ParameterOptimizer {

    private static final Logger log = LoggerFactory.getLogger(ParameterOptimizer.class);

    /*
     * ВАЖНОЕ ПРИМЕЧАНИЕ:
     * После введения в TradingBot логики с двумя режимами волатильности (High/Low),
     * данный оптимизатор стал менее эффективным. Сейчас он оптимизирует только ОДИН
     * набор параметров, который передается в конструктор TradingBot, но фактически
     * НЕ ИСПОЛЬЗУЕТСЯ внутри симуляции, так как бот берет параметры из Config.java.
     *
     * Чтобы корректно оптимизировать систему, необходимо:
     * 1. Решить, параметры какого режима мы оптимизируем (например, только для High Volatility).
     * 2. Модифицировать TradingBot, чтобы он использовал ОДИН набор оптимизируемых параметров
     * для выбранного режима, а для второго режима брал статические значения из Config.
     * 3. В идеале — использовать более продвинутые методы оптимизации (например, с помощью opt4j),
     * которые могут работать со всем набором из 8 параметров (по 4 для каждого режима) одновременно.
     *
     * Текущий код оставлен для демонстрации принципа, но требует доработки для полноценной
     * оптимизации новой, двухрежимной стратегии.
     */
    public static void main(String[] args) throws IOException {
        log.info("--- Подготовка данных для оптимизации ---");
        ProcessedData data = DataManager.prepareData();
        if (data == null) {
            log.error("Не удалось подготовить данные. Оптимизация прервана.");
            return;
        }

        File bestModelFile = new File(Config.MODEL_DIR, "bestModel.bin");
        if (!bestModelFile.exists()) {
            log.error("Файл лучшей модели не найден: {}. Оптимизация невозможна.", bestModelFile.getAbsolutePath());
            return;
        }
        MultiLayerNetwork bestModel = ModelSerializer.restoreMultiLayerNetwork(bestModelFile);
        log.info("Загружена модель для оптимизации: {}", bestModelFile.getName());

        // Эти диапазоны будут использоваться для оптимизации "базовых" параметров
        // в объекте BotParameters, хотя они и не влияют на симуляцию напрямую.
        double atrStopStart = 1.5, atrStopEnd = 5.0, atrStopStep = 0.5;
        double rrStart = 1.0, rrEnd = 3.0, rrStep = 0.25;
        double thresholdStart = 0.3, thresholdEnd = 1.5, thresholdStep = 0.2;
        double riskPercentStart = 0.02, riskPercentEnd = 0.10, riskPercentStep = 0.02;

        optimize(bestModel, data.testFeatures(),
                atrStopStart, atrStopEnd, atrStopStep,
                rrStart, rrEnd, rrStep,
                thresholdStart, thresholdEnd, thresholdStep,
                riskPercentStart, riskPercentEnd, riskPercentStep);
    }

    private static void optimize(MultiLayerNetwork model, INDArray testFeatures,
                                 double atrStopStart, double atrStopEnd, double atrStopStep,
                                 double rrStart, double rrEnd, double rrStep,
                                 double thresholdStart, double thresholdEnd, double thresholdStep,
                                 double riskStart, double riskEnd, double riskStep) {

        log.info("\n--- НАЧАЛО ОПТИМИЗАЦИИ ПАРАМЕТРОВ ---");
        log.warn("ВНИМАНИЕ: Оптимизатор запущен в демонстрационном режиме. Он не влияет на параметры, используемые в симуляции.");

        BotParameters bestParams = null;
        double bestBalance = -Double.MAX_VALUE;
        int totalIterations = 0;

        for (double risk = riskStart; risk <= riskEnd; risk += riskStep) {
            for (double atrStop = atrStopStart; atrStop <= atrStopEnd; atrStop += atrStopStep) {
                for (double rr = rrStart; rr <= rrEnd; rr += rrStep) {
                    for (double threshold = thresholdStart; threshold <= thresholdEnd; threshold += thresholdStep) {
                        totalIterations++;
                        // Создаем объект параметров, который сейчас является "пустышкой" для симуляции
                        BotParameters currentParams = new BotParameters(atrStop, rr, threshold, risk);
                        TradingBot bot = new TradingBot(currentParams, false);

                        TradingBot.SimulationResult result = bot.runSimulation(model, testFeatures);

                        if (result == null || result.totalTrades() < 10) continue;

                        if (result.finalBalance() > bestBalance) {
                            bestBalance = result.finalBalance();
                            bestParams = currentParams; // Сохраняем "демо" параметры
                            log.info(String.format("Найден новый лидер (на основе параметров из Config.java): Баланс: %.2f | Сделок: %d | ПФ: %.2f",
                                    bestBalance, result.totalTrades(), result.profitFactor()));
                        }
                    }
                }
            }
        }

        log.info("\n--- ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ---");
        log.info("Всего итераций: {}", totalIterations);
        if (bestParams != null) {
            log.info("Лучший результат (на основе статических параметров из Config): Баланс = {}", String.format("%.2f", bestBalance));
            log.info("Параметры, использованные в последнем лучшем запуске оптимизатора (демонстрационные): {}", bestParams);
        } else {
            log.warn("Не удалось найти оптимальные параметры. Проверьте логику или расширьте диапазоны.");
        }
    }
}
