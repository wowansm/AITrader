package org.varavin;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.varavin.entity.BotParameters;

import java.io.File;
import java.io.IOException;

public class NeuralNetwork {
    private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

    public static void main(String[] args) {
        try {
            setupEnvironment();
            new File(Config.MODEL_DIR).mkdirs();

            DataSetIterator[] data = DataManager.prepareData(Config.BATCH_SIZE);
            if (data == null) {
                throw new RuntimeException("Не удалось подготовить данные");
            }

            DataSetIterator trainIter = data[0];
            DataSetIterator valIter = data[1];
            DataSetIterator testIter = data[2];

            if (Config.IS_TRAINING_MODE) {
                MultiLayerNetwork model = createOrLoadModel();
                trainModel(model, trainIter, valIter);
            }

            runBacktestAndEvaluation(testIter);

        } catch (Exception e) {
            log.error("Критическая ошибка: ", e);
            System.exit(1);
        }
    }

    private static void setupEnvironment() {
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        log.info("ND4J Data Type: {}", Nd4j.dataType());
    }

    private static MultiLayerNetwork createOrLoadModel() throws IOException {
        File bestModelFile = new File(Config.MODEL_DIR, "bestModel.bin");
        if (Config.LOAD_EXISTING_MODEL_FOR_TRAINING && bestModelFile.exists()) {
            log.info("Загрузка существующей лучшей модели для продолжения обучения...");
            return ModelSerializer.restoreMultiLayerNetwork(bestModelFile, true);
        }

        log.info("Создание новой модели типа: {}", Config.MODEL_TYPE);
        MultiLayerConfiguration config = createRegressionModel();


        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("Создана новая модель с {} параметрами", model.numParams());
        log.info(model.summary());
        return model;
    }

    // --- ИЗМЕНЕНИЕ: Убрали SelfAttentionLayer ---
    private static MultiLayerConfiguration createRegressionModel() {
        Adam adamUpdater = new Adam(new ExponentialSchedule(ScheduleType.EPOCH, Config.INITIAL_LEARNING_RATE, Config.LR_DECAY_RATE));

        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(adamUpdater)
                .l2(Config.L2_REGULARIZATION)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .list()
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(Config.CNN_KERNEL_SIZE)
                        .stride(1)
                        .nIn(Config.NUM_FEATURES)
                        .nOut(Config.CNN_N_FILTERS)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new LSTM.Builder()
                        .nIn(Config.CNN_N_FILTERS)
                        .nOut(Config.LSTM_LAYER_SIZE)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .build())
                .layer(new GlobalPoolingLayer.Builder().build())
                .layer(new DenseLayer.Builder()
                        .nIn(Config.LSTM_LAYER_SIZE)
                        .nOut(Config.DENSE_LAYER_SIZE)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DropoutLayer(Config.DROPOUT_RATE))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .nIn(Config.DENSE_LAYER_SIZE)
                        .nOut(Config.NUM_OUTPUTS)
                        .activation(Activation.IDENTITY)
                        .build())
                .setInputType(InputType.recurrent(Config.NUM_FEATURES, Config.TIME_STEPS))
                .dataType(DataType.FLOAT)
                .build();
    }

    private static void trainModel(MultiLayerNetwork model, DataSetIterator trainIter, DataSetIterator valIter) {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(100)
        );

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(
                        new MaxEpochsTerminationCondition(Config.MAX_EPOCHS),
                        new ScoreImprovementEpochTerminationCondition(Config.EARLY_STOPPING_PATIENCE, 1e-5)
                )
                .scoreCalculator(new DataSetLossCalculator(valIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(Config.MODEL_DIR))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        try {
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
            log.info("Обучение завершено.");
            log.info("Причина остановки: {}", result.getTerminationReason());
            log.info("Лучшая эпоха: {}", result.getBestModelEpoch());
        } catch (Exception e) {
            log.error("Обучение прервано из-за исключения", e);
        }
    }

    private static void runBacktestAndEvaluation(DataSetIterator testIter) throws IOException {
        File bestModelFile = new File(Config.MODEL_DIR, "bestModel.bin");
        if (!bestModelFile.exists()) {
            log.error("Файл лучшей модели не найден для оценки и бэктеста!");
            return;
        }

        MultiLayerNetwork bestModel = ModelSerializer.restoreMultiLayerNetwork(bestModelFile);
        log.info("Загружена лучшая модель: {}", bestModelFile.getName());

        log.info("--- Оценка качества регрессии на тестовых данных ---");
        RegressionEvaluation eval = bestModel.evaluateRegression(testIter);
        log.info("Статистика по выходам (Column 0 = Pred_Up_ATR, Column 1 = Pred_Down_ATR):");
        log.info(eval.stats());

        testIter.reset();

        log.info("\n--- Запуск бэктеста на тестовых данных ---");

        BotParameters botParameters = new BotParameters(
                Config.ATR_STOP_MULTIPLIER,
                Config.FIXED_RISK_REWARD_RATIO,
                Config.SIGNAL_THRESHOLD,
                Config.RISK_PER_TRADE_PERCENT
        );
        TradingBot bot = new TradingBot(botParameters, true);
        bot.runSimulation(bestModel, testIter);
    }
}
