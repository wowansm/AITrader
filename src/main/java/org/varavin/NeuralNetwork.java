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
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions; // ИЗМЕНЕНИЕ: Используем стандартный LossFunctions
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.varavin.entity.BatchDataSetIterator;
import org.varavin.entity.BotParameters;
import org.varavin.entity.ProcessedData;

import java.io.File;
import java.io.IOException;

public class NeuralNetwork {
    private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

    public static void main(String[] args) {
        try {
            setupEnvironment();
            new File(Config.MODEL_DIR).mkdirs();

            ProcessedData data = DataManager.prepareData();
            if (data == null) throw new RuntimeException("Не удалось подготовить данные");

            if (Config.IS_TRAINING_MODE) {
                // ИЗМЕНЕНИЕ: trainLabels больше не нужен для создания модели
                MultiLayerNetwork model = createOrLoadModel();
                DataSetIterator trainIter = new BatchDataSetIterator(data.trainFeatures(), data.trainLabels(), Config.BATCH_SIZE);
                DataSetIterator valIter = new BatchDataSetIterator(data.valFeatures(), data.valLabels(), Config.BATCH_SIZE);
                trainModel(model, trainIter, valIter);
            }

            runBacktestAndEvaluation(data);

        } catch (Exception e) {
            log.error("Критическая ошибка: ", e);
            System.exit(1);
        }
    }

    private static void setupEnvironment() {
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        log.info("ND4J Data Type: {}", Nd4j.dataType());
    }

    // ИЗМЕНЕНИЕ: Метод больше не принимает trainLabels
    private static MultiLayerNetwork createOrLoadModel() throws IOException {
        File bestModelFile = new File(Config.MODEL_DIR, "bestModel.bin");
        if (Config.LOAD_EXISTING_MODEL_FOR_TRAINING && bestModelFile.exists()) {
            log.info("Загрузка существующей лучшей модели для продолжения обучения...");
            return ModelSerializer.restoreMultiLayerNetwork(bestModelFile, true);
        }

        log.info("Создание новой модели типа: CNN");
        MultiLayerConfiguration config = createClassificationModel();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("Создана новая модель с {} параметрами", model.numParams());
        log.info(model.summary());
        return model;
    }

    // ИЗМЕНЕНИЕ: Метод больше не принимает trainLabels
    private static MultiLayerConfiguration createClassificationModel() {
        Adam adamUpdater = new Adam(new StepSchedule(
                ScheduleType.EPOCH,
                Config.INITIAL_LEARNING_RATE,
                Config.LR_SCHEDULE_DECAY_RATE,
                Config.LR_SCHEDULE_STEP
        ));

        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(adamUpdater)
                .l2(Config.L2_REGULARIZATION)
                .list()
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(5)
                        .nOut(Config.Convolution_LAYER_SIZE1)
                        .stride(1)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new Subsampling1DLayer.Builder(PoolingType.MAX)
                        .kernelSize(2)
                        .stride(2)
                        .build())
                .layer(new Convolution1DLayer.Builder()
                        .kernelSize(3)
                        .nOut(Config.Convolution_LAYER_SIZE2)
                        .stride(1)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                .layer(new DenseLayer.Builder()
                        .nIn(Config.Convolution_LAYER_SIZE2)
                        .nOut(Config.DENSE_LAYER_SIZE)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DropoutLayer(Config.DROPOUT_RATE))
                // ИЗМЕНЕНИЕ: Используем стандартную функцию потерь, так как данные уже сбалансированы
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(Config.DENSE_LAYER_SIZE)
                        .nOut(Config.NUM_OUTPUTS)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.recurrent(Config.NUM_FEATURES, Config.TIME_STEPS))
                .dataType(DataType.FLOAT)
                .build();
    }

    // ИЗМЕНЕНИЕ: Метод calculateClassWeights больше не нужен

    private static void trainModel(MultiLayerNetwork model, DataSetIterator trainIter, DataSetIterator valIter) {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(Config.MAX_EPOCHS), new ScoreImprovementEpochTerminationCondition(Config.EARLY_STOPPING_PATIENCE, 1e-5))
                .scoreCalculator(new DataSetLossCalculator(valIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(Config.MODEL_DIR))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        try {
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
            log.info("Обучение завершено. Причина: {}, Лучшая эпоха: {}", result.getTerminationReason(), result.getBestModelEpoch());
        } catch (Exception e) {
            log.error("Обучение прервано из-за исключения", e);
        }
    }

    private static void runBacktestAndEvaluation(ProcessedData data) throws IOException {
        File bestModelFile = new File(Config.MODEL_DIR, "bestModel.bin");
        if (!bestModelFile.exists()) {
            log.error("Файл лучшей модели не найден для оценки и бэктеста!");
            return;
        }

        MultiLayerNetwork bestModel = ModelSerializer.restoreMultiLayerNetwork(bestModelFile);
        log.info("Загружена лучшая модель: {}", bestModelFile.getName());

        log.info("--- Оценка качества классификации на тестовых данных ---");

        DataSetIterator testIter = new BatchDataSetIterator(data.testFeatures(), data.testLabels(), Config.BATCH_SIZE);
        Evaluation eval = bestModel.evaluate(testIter);

        log.info("Статистика по классам (0=UP, 1=DOWN, 2=SIDEWAYS):");
        log.info(eval.stats());

        log.info("\n--- Запуск бэктеста на тестовых данных ---");
        BotParameters botParameters = new BotParameters(Config.ATR_STOP_MULTIPLIER, Config.FIXED_RISK_REWARD_RATIO, Config.SIGNAL_THRESHOLD, Config.RISK_PER_TRADE_PERCENT);
        TradingBot bot = new TradingBot(botParameters, true);
        bot.runSimulation(bestModel, data.testFeatures());
    }
}
