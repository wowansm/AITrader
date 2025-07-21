package org.varavin;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ta4j.core.*;
import org.ta4j.core.indicators.ATRIndicator;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.adx.ADXIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.num.DoubleNum;
import org.varavin.entity.BotParameters;

public class TradingBot {
    private static final Logger log = LoggerFactory.getLogger(TradingBot.class);

    public record SimulationResult(double finalBalance, double profitFactor, int totalTrades) {}

    private enum VolatilityRegime { HIGH, LOW }
    private enum PositionType { LONG, SHORT }

    private static class Position {
        private static int nextId = 1;
        final int id;
        final PositionType type;
        double entryPrice;
        int quantity;
        final double stopLossPrice;
        final double takeProfitPrice;

        Position(PositionType type, double entryPrice, int quantity, double stopLossPrice, double takeProfitPrice) {
            this.id = nextId++;
            this.type = type;
            this.entryPrice = entryPrice;
            this.quantity = quantity;
            this.stopLossPrice = stopLossPrice;
            this.takeProfitPrice = takeProfitPrice;
        }
    }

    private enum SignalType {
        LONG, SHORT, NONE
    }

    private final BotParameters params;
    private final double initialBalance;
    private double currentBalance;
    private Position currentPosition;
    private final boolean needPrintLog;

    private int totalTrades = 0;
    private int winTrades = 0;
    private double grossProfit = 0.0;
    private double grossLoss = 0.0;

    public TradingBot(BotParameters params, boolean needPrintLog) {
        this.params = params;
        this.initialBalance = Config.INITIAL_BALANCE;
        this.currentBalance = initialBalance;
        this.currentPosition = null;
        this.needPrintLog = needPrintLog;
    }

    public SimulationResult runSimulation(MultiLayerNetwork model, INDArray testFeatures) {
        resetState();

        BarSeries originalSeries = DataManager.getOriginalSeries();
        int testDataStartIndex = DataManager.getTestDataStartIndex();

        if (originalSeries == null || testDataStartIndex < 0) {
            log.error("Не удалось получить исходные данные для симуляции!");
            return new SimulationResult(initialBalance, 0, 0);
        }

        INDArray allPredictions = model.output(testFeatures);
        BarSeries simulationSeries = new BaseBarSeriesBuilder().withNumTypeOf(DoubleNum.class).withName("SIMULATION_SERIES").build();
        for (int i = 0; i < testDataStartIndex; i++) {
            simulationSeries.addBar(originalSeries.getBar(i));
        }

        for (int step = 0; step < testFeatures.size(0); step++) {
            int originalIndex = testDataStartIndex + step;
            if (originalIndex >= originalSeries.getBarCount()) break;

            Bar currentBar = originalSeries.getBar(originalIndex);
            simulationSeries.addBar(currentBar);
            int seriesEndIndex = simulationSeries.getEndIndex();

            ClosePriceIndicator closePrice = new ClosePriceIndicator(simulationSeries);
            EMAIndicator emaFilter = new EMAIndicator(closePrice, 200);
            ADXIndicator adxFilter = new ADXIndicator(simulationSeries, 14);
            ATRIndicator atrIndicator = new ATRIndicator(simulationSeries, 14);

            SMAIndicator longTermAtr = new SMAIndicator(atrIndicator, Config.REGIME_FILTER_PERIOD);
            double currentAtr = atrIndicator.getValue(seriesEndIndex).doubleValue();
            double avgAtr = longTermAtr.getValue(seriesEndIndex).doubleValue();
            VolatilityRegime currentRegime = (currentAtr > avgAtr * Config.REGIME_VOLATILITY_THRESHOLD) ? VolatilityRegime.HIGH : VolatilityRegime.LOW;

            double currentPrice = currentBar.getClosePrice().doubleValue();
            if (currentPrice <= 0 || currentAtr <= 0) continue;

            if (isPositionOpen()) {
                checkExits(currentPrice);
            }

            if (!isPositionOpen()) {
                INDArray currentPrediction = allPredictions.getRow(step);
                double emaValue = emaFilter.getValue(seriesEndIndex).doubleValue();
                double adxValue = adxFilter.getValue(seriesEndIndex).doubleValue();
                handleNoPosition(currentPrice, currentAtr, currentPrediction, emaValue, adxValue, currentRegime);
            }
        }

        if (isPositionOpen()) {
            double finalPrice = simulationSeries.getLastBar().getClosePrice().doubleValue();
            if (finalPrice > 0) closePosition(finalPrice, "Конец симуляции");
        }

        if (needPrintLog) printResults();

        double profitFactor = (grossLoss > 0) ? grossProfit / grossLoss : Double.POSITIVE_INFINITY;
        return new SimulationResult(currentBalance, profitFactor, totalTrades);
    }

    private void resetState() {
        this.currentBalance = this.initialBalance;
        this.currentPosition = null;
        this.totalTrades = 0;
        this.winTrades = 0;
        this.grossProfit = 0.0;
        this.grossLoss = 0.0;
        Position.nextId = 1;
    }

    private void handleNoPosition(double currentPrice, double currentAtr, INDArray prediction, double emaValue, double adxValue, VolatilityRegime regime) {
        double signalThreshold = (regime == VolatilityRegime.HIGH) ? Config.SIGNAL_THRESHOLD_HV : Config.SIGNAL_THRESHOLD_LV;
        SignalType signal = getSignal(prediction, signalThreshold);

        if (signal != SignalType.NONE) {
            boolean isTrendConfirmed = adxValue > Config.ADX_TREND_THRESHOLD;
            if (isTrendConfirmed) {
                if (signal == SignalType.LONG && currentPrice > emaValue) {
                    enterPosition(currentPrice, currentAtr, SignalType.LONG, regime);
                } else if (signal == SignalType.SHORT && Config.ALLOW_SHORT_TRADING && currentPrice < emaValue) {
                    enterPosition(currentPrice, currentAtr, SignalType.SHORT, regime);
                }
            }
        }
    }

    private boolean checkExits(double currentPrice) {
        if (!isPositionOpen()) return false;
        PositionType type = currentPosition.type;
        if ((type == PositionType.LONG && currentPrice <= currentPosition.stopLossPrice) || (type == PositionType.SHORT && currentPrice >= currentPosition.stopLossPrice)) {
            closePosition(currentPosition.stopLossPrice, "Stop-Loss");
            return true;
        }
        if ((type == PositionType.LONG && currentPrice >= currentPosition.takeProfitPrice) || (type == PositionType.SHORT && currentPrice <= currentPosition.takeProfitPrice)) {
            closePosition(currentPosition.takeProfitPrice, "Take-Profit");
            return true;
        }
        return false;
    }

    private void enterPosition(double price, double atrValue, SignalType signalType, VolatilityRegime regime) {
        double atrStopMultiplier = (regime == VolatilityRegime.HIGH) ? Config.ATR_STOP_MULTIPLIER_HV : Config.ATR_STOP_MULTIPLIER_LV;
        double riskRewardRatio = (regime == VolatilityRegime.HIGH) ? Config.FIXED_RISK_REWARD_RATIO_HV : Config.FIXED_RISK_REWARD_RATIO_LV;

        double stopLossDistance = atrValue * atrStopMultiplier;
        if (stopLossDistance <= 1e-6) return;

        double takeProfitDistance = stopLossDistance * riskRewardRatio;
        double stopLossPrice, takeProfitPrice;

        if (signalType == SignalType.LONG) {
            stopLossPrice = price - stopLossDistance;
            takeProfitPrice = price + takeProfitDistance;
        } else {
            stopLossPrice = price + stopLossDistance;
            takeProfitPrice = price - takeProfitDistance;
        }

        int lotSize = calculatePositionSize(price, stopLossPrice, regime);
        if (lotSize < Config.MINIMUM_LOT_SIZE) return;

        currentPosition = new Position(signalType == SignalType.LONG ? PositionType.LONG : PositionType.SHORT, price, lotSize, stopLossPrice, takeProfitPrice);
        double commission = lotSize * price * Config.COMMISSION_RATE;
        currentBalance -= commission;

        if(needPrintLog) {
            log.info("OPEN {} (Regime: {}, Pos #{}, Price {}): {} @ {} | TP: {} | SL: {}",
                    currentPosition.type, regime, currentPosition.id, String.format("%.2f", price), currentPosition.quantity, String.format("%.2f", price),
                    String.format("%.2f", takeProfitPrice), String.format("%.2f", stopLossPrice));
        }
    }

    private int calculatePositionSize(double entryPrice, double stopLossPrice, VolatilityRegime regime) {
        double riskPercent = (regime == VolatilityRegime.HIGH) ? Config.RISK_PER_TRADE_PERCENT_HV : Config.RISK_PER_TRADE_PERCENT_LV;

        double riskPerShare = Math.abs(entryPrice - stopLossPrice);
        if (riskPerShare <= 1e-6) return 0;

        double riskCapital = currentBalance * riskPercent;
        int quantityByRisk = (int) (riskCapital / riskPerShare);
        int quantityByBalance = (int) (currentBalance / (entryPrice * (1 + Config.COMMISSION_RATE)));
        int desiredQuantity = Math.min(quantityByRisk, quantityByBalance);

        return (desiredQuantity / Config.MINIMUM_LOT_SIZE) * Config.MINIMUM_LOT_SIZE;
    }

    private void closePosition(double price, String reason) {
        if (!isPositionOpen()) return;

        double entryValue = currentPosition.quantity * currentPosition.entryPrice;
        double exitValue = currentPosition.quantity * price;
        double exitCommission = exitValue * Config.COMMISSION_RATE;

        double pnl = (currentPosition.type == PositionType.LONG) ? (exitValue - entryValue) : (entryValue - exitValue);
        double netProfit = pnl - exitCommission;

        currentBalance += netProfit;
        totalTrades++;

        if (netProfit > 0) {
            winTrades++;
            grossProfit += netProfit;
        } else {
            grossLoss += Math.abs(netProfit);
        }

        if(needPrintLog) {
            log.info("CLOSE {} (Pos #{}, {}): {} @ {} | Net Profit: {} | Balance: {}",
                    currentPosition.type, currentPosition.id, reason, currentPosition.quantity,
                    String.format("%.2f", price), String.format("%.2f", netProfit), String.format("%.2f", currentBalance));
        }
        currentPosition = null;
    }

    private boolean isPositionOpen() {
        return currentPosition != null;
    }

    private SignalType getSignal(INDArray prediction, double signalThreshold) {
        // --- ИЗМЕНЕНИЕ: Исправлен вызов argMax для 1D вектора ---
        // Убран второй аргумент (ось), так как prediction - это 1D вектор.
        int predictedClass = Nd4j.argMax(prediction).getInt(0);
        double probability = prediction.getDouble(predictedClass);

        if (probability < signalThreshold) {
            return SignalType.NONE;
        }

        return switch (predictedClass) {
            case 0 -> SignalType.LONG;
            case 1 -> SignalType.SHORT;
            default -> SignalType.NONE;
        };
    }

    private void printResults() {
        double profit = currentBalance - initialBalance;
        double profitPercent = (initialBalance > 0) ? (profit / initialBalance) * 100 : 0.0;
        double winRate = totalTrades > 0 ? ((double) winTrades / totalTrades) * 100 : 0;
        double profitFactor = (grossLoss > 0) ? grossProfit / grossLoss : Double.POSITIVE_INFINITY;

        log.info("\n--- Результаты симуляции ---");
        log.info("Начальный баланс:   {}", String.format("%.2f", initialBalance));
        log.info("Конечный баланс:    {}", String.format("%.2f", currentBalance));
        log.info("Прибыль/убыток:     {} ({}%)", String.format("%.2f", profit), String.format("%.2f", profitPercent));
        log.info("Всего сделок:       {}", totalTrades);
        log.info("Прибыльных сделок:  {} ({}%)", winTrades, String.format("%.2f", winRate));
        log.info("Профит-фактор:      {}", profitFactor == Double.POSITIVE_INFINITY ? "Infinity" : String.format("%.2f", profitFactor));
        log.info("----------------------------");
    }
}
