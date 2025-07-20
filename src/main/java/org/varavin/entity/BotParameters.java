package org.varavin.entity;

/**
 * Хранит набор торговых параметров для одной симуляции или оптимизации.
 * Этот record является публичным и доступен для всех классов в проекте.
 */
public record BotParameters(
        double atrStopMultiplier,      // Множитель ATR для стоп-лосса
        double fixedRiskRewardRatio, // Фиксированный R/R для тейк-профита
        double signalThreshold,      // Минимальная сила сигнала (в ATR) для входа
        double riskPercent           // Процент риска на сделку
) {}

