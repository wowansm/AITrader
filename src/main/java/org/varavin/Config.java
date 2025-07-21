package org.varavin;

public class Config {

    // --- Параметры данных и кэширования ---
    public static final String CSV_FILE_NAME = "candles_gazprom_5m.csv";
    public static final String CACHE_DIR = "data_cache_trading";
    // --- ИЗМЕНЕНИЕ: Новая версия данных для простого набора признаков ---
    public static final String DATA_VERSION = "v14_final_features";

    // --- Выбор типа модели ---
    public static final String MODEL_TYPE = "CLASSIFICATION";

    // --- Параметры признаков и временных рядов ---
    public static final int TIME_STEPS = 30;
    // --- ИЗМЕНЕНИЕ: Уменьшаем количество признаков до 9 ---
    public static final int NUM_FEATURES = 18;
    public static final int NUM_OUTPUTS = 3;
    public static final int MAX_FUTURE_TICKS = 12;
    public static final int MAX_INDICATOR_PERIOD = 200;
    // --- ИЗМЕНЕНИЕ: Параметры для создания меток классификации ---
    public static final double CLASSIFICATION_ATR_UP_THRESHOLD = 1.5;   // Движение вверх на 1.5 ATR для метки UP
    public static final double CLASSIFICATION_ATR_DOWN_THRESHOLD = 1.5; // Движение вниз на 1.5 ATR для метки DOWN

    // --- Параметры Нейронной сети ---
    public static final String MODEL_DIR = "models_trading_classification";
    public static final boolean IS_TRAINING_MODE = true; // Установите true для переобучения
    public static final boolean LOAD_EXISTING_MODEL_FOR_TRAINING = false;
    public static final double INITIAL_LEARNING_RATE = 3e-4; // Умеренная начальная скорость
    public static final double L2_REGULARIZATION = 2e-4;
    public static final double DROPOUT_RATE = 0.4;
    public static final int LSTM_LAYER_SIZE = 100;
    public static final int DENSE_LAYER_SIZE = 100;
    public static final int Convolution_LAYER_SIZE1 = 80;
    public static final int Convolution_LAYER_SIZE2 = 100;
    public static final int BATCH_SIZE = 64;
    public static final int EARLY_STOPPING_PATIENCE = 30;
    public static final int MAX_EPOCHS = 300;
    public static final int CNN_N_FILTERS = 48;
    public static final int CNN_KERNEL_SIZE = 5;
    public static final double LR_DECAY_RATE = 0.98;
    public static final double LR_SCHEDULE_DECAY_RATE = 0.5; // Снижать в 2 раза
    public static final int LR_SCHEDULE_STEP = 15;           // Каждые 15 эпох

    // --- Параметры Торгового Робота (остаются без изменений на данном этапе) ---
    public static final double INITIAL_BALANCE = 50000.0;
    public static final double COMMISSION_RATE = 0.0004;
    public static final int MINIMUM_LOT_SIZE = 10;
    public static final boolean ALLOW_SHORT_TRADING = true;
    public static final double ADX_TREND_THRESHOLD = 20.0;

    // --- ПАРАМЕТРЫ ДЛЯ ФИНАЛЬНОГО БЭКТЕСТА (остаются без изменений) ---
    public static final double RISK_PER_TRADE_PERCENT = 0.02;
    public static final double ATR_STOP_MULTIPLIER = 4.0;
    public static final double FIXED_RISK_REWARD_RATIO = 2.25;
    public static final double SIGNAL_THRESHOLD = 0.55; // Для классификации порог обычно 0.5

    // --- Параметры для фильтра режимов волатильности (остаются без изменений) ---
    public static final int REGIME_FILTER_PERIOD = 100;
    public static final double REGIME_VOLATILITY_THRESHOLD = 1.25;
    public static final double RISK_PER_TRADE_PERCENT_HV = 0.03;
    public static final double ATR_STOP_MULTIPLIER_HV = 4.0;
    public static final double FIXED_RISK_REWARD_RATIO_HV = 2.5;
    public static final double SIGNAL_THRESHOLD_HV = 0.5;
    public static final double RISK_PER_TRADE_PERCENT_LV = 0.05;
    public static final double ATR_STOP_MULTIPLIER_LV = 2.0;
    public static final double FIXED_RISK_REWARD_RATIO_LV = 1.5;
    public static final double SIGNAL_THRESHOLD_LV = 0.6;
}
