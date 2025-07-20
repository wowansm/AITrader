package org.varavin;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ta4j.core.BarSeries;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.adx.ADXIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.varavin.entity.BotParameters;

import java.util.ArrayList;
import java.util.List;

public class TradingBot {
    private static final Logger log = LoggerFactory.getLogger(TradingBot.class);

    // Этот record виден из ParameterOptimizer, так как он public
    public record SimulationResult(double finalBalance, double profitFactor, int totalTrades) {}

    private enum PositionType {
        LONG, SHORT
    }

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

    private final EMAIndicator emaFilter;
    private final ADXIndicator adxFilter;

    // Этот конструктор соответствует тому, что ожидают ParameterOptimizer и NeuralNetwork
    public TradingBot(BotParameters params, boolean needPrintLog) {
        this.params = params;
        this.initialBalance = Config.INITIAL_BALANCE;
        this.currentBalance = initialBalance;
        this.currentPosition = null;
        this.needPrintLog = needPrintLog;

        BarSeries series = DataManager.getOriginalSeries();
        if (series != null) {
            this.emaFilter = new EMAIndicator(new ClosePriceIndicator(series), 200);
            this.adxFilter = new ADXIndicator(series, 14);
        } else {
            this.emaFilter = null;
            this.adxFilter = null;
            log.error("Не удалось получить BarSeries для инициализации фильтров!");
        }
    }

    public SimulationResult runSimulation(MultiLayerNetwork model, DataSetIterator testIterator) {
        if (this.emaFilter == null || this.adxFilter == null) {
            return null;
        }

        resetState();

        testIterator.reset();
        int step = 0;

        List<DataSet> testData = new ArrayList<>();
        testIterator.forEachRemaining(testData::add);
        INDArray allFeatures = Nd4j.vstack(testData.stream().map(DataSet::getFeatures).toArray(INDArray[]::new));
        INDArray allPredictions = model.output(allFeatures);

        while (step < allFeatures.size(0)) {
            double currentPrice = DataManager.getOriginalPrice(step);
            if (currentPrice <= 0) {
                step++;
                continue;
            }

            if (isPositionOpen()) {
                if (checkExits(currentPrice)) {
                    step++;
                    continue;
                }
            }

            if (!isPositionOpen()) {
                INDArray currentPrediction = allPredictions.getRow(step);
                handleNoPosition(currentPrice, step, currentPrediction);
            }

            step++;
        }

        if (isPositionOpen()) {
            double finalPrice = DataManager.getOriginalPrice(step - 1);
            if (finalPrice > 0) {
                closePosition(finalPrice, "Конец симуляции");
            }
        }

        if (needPrintLog) {
            printResults();
        }

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

    private void handleNoPosition(double currentPrice, int step, INDArray prediction) {
        int originalIndex = DataManager.getTestDataStartIndex() + step;

        SignalType signal = getSignal(prediction);

        if (signal != SignalType.NONE) {
            boolean isTrendConfirmed = adxFilter.getValue(originalIndex).doubleValue() > Config.ADX_TREND_THRESHOLD;

            if (isTrendConfirmed) {
                if (signal == SignalType.LONG && currentPrice > emaFilter.getValue(originalIndex).doubleValue()) {
                    enterPosition(currentPrice, step, SignalType.LONG);
                } else if (signal == SignalType.SHORT && Config.ALLOW_SHORT_TRADING && currentPrice < emaFilter.getValue(originalIndex).doubleValue()) {
                    enterPosition(currentPrice, step, SignalType.SHORT);
                }
            }
        }
    }

    private boolean checkExits(double currentPrice) {
        if (!isPositionOpen()) return false;

        if (currentPosition.type == PositionType.LONG) {
            if (currentPrice <= currentPosition.stopLossPrice) {
                closePosition(currentPosition.stopLossPrice, "Stop-Loss");
                return true;
            }
            if (currentPrice >= currentPosition.takeProfitPrice) {
                closePosition(currentPosition.takeProfitPrice, "Take-Profit");
                return true;
            }
        } else { // SHORT
            if (currentPrice >= currentPosition.stopLossPrice) {
                closePosition(currentPosition.stopLossPrice, "Stop-Loss");
                return true;
            }
            if (currentPrice <= currentPosition.takeProfitPrice) {
                closePosition(currentPosition.takeProfitPrice, "Take-Profit");
                return true;
            }
        }
        return false;
    }

    private void enterPosition(double price, int step, SignalType signalType) {
        double atrValue = DataManager.getOriginalAtr(step);
        if (atrValue <= 0) return;

        double stopLossDistance = atrValue * params.atrStopMultiplier();
        if (stopLossDistance <= 1e-6) return;

        double takeProfitDistance = stopLossDistance * params.fixedRiskRewardRatio();

        double stopLossPrice, takeProfitPrice;

        if (signalType == SignalType.LONG) {
            stopLossPrice = price - stopLossDistance;
            takeProfitPrice = price + takeProfitDistance;
        } else { // SHORT
            stopLossPrice = price + stopLossDistance;
            takeProfitPrice = price - takeProfitDistance;
        }

        int lotSize = calculatePositionSize(price, stopLossPrice);
        if (lotSize < Config.MINIMUM_LOT_SIZE) {
            return;
        }

        currentPosition = new Position(signalType == SignalType.LONG ? PositionType.LONG : PositionType.SHORT, price, lotSize, stopLossPrice, takeProfitPrice);
        double commission = lotSize * price * Config.COMMISSION_RATE;
        currentBalance -= commission;

        if(needPrintLog) {
            log.info("OPEN {} (Pos #{}, Step {}): {} @ {} | TP: {} | SL: {}",
                    currentPosition.type, currentPosition.id, step, currentPosition.quantity, String.format("%.2f", price),
                    String.format("%.2f", takeProfitPrice), String.format("%.2f", stopLossPrice));
        }
    }

    private int calculatePositionSize(double entryPrice, double stopLossPrice) {
        double riskPerShare = Math.abs(entryPrice - stopLossPrice);
        if (riskPerShare <= 1e-6) return 0;

        double riskCapital = currentBalance * params.riskPercent();
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

    private SignalType getSignal(INDArray prediction) {
        double predUpMoveInAtr = prediction.getDouble(0);
        double predDownMoveInAtr = prediction.getDouble(1);

        boolean longSignal = predUpMoveInAtr > predDownMoveInAtr && predUpMoveInAtr > params.signalThreshold();
        boolean shortSignal = predDownMoveInAtr > predUpMoveInAtr && predDownMoveInAtr > params.signalThreshold();

        if (longSignal) return SignalType.LONG;
        if (shortSignal) return SignalType.SHORT;
        return SignalType.NONE;
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
