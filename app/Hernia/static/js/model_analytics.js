// Model Analytics - Análisis avanzado de modelos
class ModelAnalytics {
    constructor() {
        this.modelData = {};
        this.classNames = ['Hernia', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'Sin Hernia', 'T12'];
    }

    setModelData(data) {
        this.modelData = data;
    }

    // Calcular métricas estadísticas avanzadas
    calculateAdvancedMetrics() {
        const results = {};
        
        Object.keys(this.modelData).forEach(key => {
            const data = this.modelData[key];
            const metrics = data.performance_metrics.per_class_metrics;
            
            // Calcular estadísticas por modelo
            const accuracies = Object.values(metrics).map(m => m.accuracy).filter(a => a > 0);
            const precisions = Object.values(metrics).map(m => m.precision).filter(p => p > 0);
            const recalls = Object.values(metrics).map(m => m.recall).filter(r => r > 0);
            const f1Scores = Object.values(metrics).map(m => m.f1_score).filter(f => f > 0);
            
            results[key] = {
                model_name: data.model_info.architecture || data.model_info.name,
                accuracy_stats: this.calculateStats(accuracies),
                precision_stats: this.calculateStats(precisions),
                recall_stats: this.calculateStats(recalls),
                f1_stats: this.calculateStats(f1Scores),
                class_balance: this.calculateClassBalance(metrics),
                convergence_metrics: this.calculateConvergenceMetrics(data),
                efficiency_score: this.calculateEfficiencyScore(data)
            };
        });
        
        return results;
    }

    calculateStats(values) {
        if (values.length === 0) return { mean: 0, std: 0, min: 0, max: 0, median: 0 };
        
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        const sorted = [...values].sort((a, b) => a - b);
        const median = sorted.length % 2 === 0 
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        
        return {
            mean: mean,
            std: std,
            min: Math.min(...values),
            max: Math.max(...values),
            median: median
        };
    }

    calculateClassBalance(metrics) {
        const totalSamples = Object.values(metrics).reduce((sum, m) => sum + m.total_samples, 0);
        const classDistribution = {};
        
        Object.keys(metrics).forEach(className => {
            const classMetric = metrics[className];
            classDistribution[className] = {
                samples: classMetric.total_samples,
                percentage: (classMetric.total_samples / totalSamples) * 100
            };
        });
        
        // Calcular índice de balance (0 = perfectamente balanceado, 1 = completamente desbalanceado)
        const expectedPercentage = 100 / Object.keys(metrics).length;
        const imbalanceScore = Object.values(classDistribution).reduce((sum, dist) => {
            return sum + Math.abs(dist.percentage - expectedPercentage);
        }, 0) / (2 * (100 - expectedPercentage));
        
        return {
            distribution: classDistribution,
            imbalance_score: imbalanceScore,
            total_samples: totalSamples
        };
    }

    calculateConvergenceMetrics(data) {
        const valLoss = data.training_results.val_loss_history;
        if (!valLoss || valLoss.length === 0) return null;
        
        // Encontrar el punto de mejor convergencia
        const bestEpoch = data.training_results.best_epoch || 0;
        const bestLoss = data.training_results.best_val_loss;
        
        // Calcular estabilidad (varianza en las últimas épocas)
        const lastEpochs = valLoss.slice(-10);
        const stability = this.calculateStats(lastEpochs).std;
        
        // Calcular velocidad de convergencia
        const initialLoss = valLoss[0];
        const convergenceRate = (initialLoss - bestLoss) / bestEpoch;
        
        // Detectar overfitting
        const trainLoss = data.training_results.train_loss_history;
        const finalTrainLoss = trainLoss[trainLoss.length - 1];
        const finalValLoss = valLoss[valLoss.length - 1];
        const overfittingGap = finalValLoss - finalTrainLoss;
        
        return {
            best_epoch: bestEpoch,
            best_loss: bestLoss,
            convergence_rate: convergenceRate,
            stability: stability,
            overfitting_gap: overfittingGap,
            total_epochs: valLoss.length
        };
    }

    calculateEfficiencyScore(data) {
        const accuracy = data.performance_metrics.overall_accuracy;
        const fps = data.speed_metrics.fps;
        const trainingTime = data.speed_metrics.training_time_minutes || 60; // Default si no está disponible
        
        // Score compuesto: accuracy ponderada por velocidad y tiempo de entrenamiento
        const speedScore = Math.min(fps / 10, 1); // Normalizar FPS (máximo 10)
        const timeScore = Math.max(0, 1 - (trainingTime / 120)); // Penalizar entrenamientos largos
        
        return {
            accuracy_score: accuracy / 100,
            speed_score: speedScore,
            time_score: timeScore,
            composite_score: (accuracy / 100) * 0.6 + speedScore * 0.3 + timeScore * 0.1
        };
    }

    // Comparar modelos y generar ranking
    generateModelRanking() {
        const metrics = this.calculateAdvancedMetrics();
        const rankings = {
            overall: [],
            accuracy: [],
            speed: [],
            efficiency: [],
            stability: []
        };
        
        Object.keys(metrics).forEach(key => {
            const metric = metrics[key];
            const modelName = metric.model_name;
            
            rankings.overall.push({
                model: modelName,
                key: key,
                score: metric.efficiency_score.composite_score
            });
            
            rankings.accuracy.push({
                model: modelName,
                key: key,
                score: this.modelData[key].performance_metrics.overall_accuracy
            });
            
            rankings.speed.push({
                model: modelName,
                key: key,
                score: this.modelData[key].speed_metrics.fps
            });
            
            rankings.efficiency.push({
                model: modelName,
                key: key,
                score: metric.efficiency_score.composite_score
            });
            
            if (metric.convergence_metrics) {
                rankings.stability.push({
                    model: modelName,
                    key: key,
                    score: 1 / (1 + metric.convergence_metrics.stability) // Invertir para que menor varianza = mejor
                });
            }
        });
        
        // Ordenar rankings
        Object.keys(rankings).forEach(category => {
            rankings[category].sort((a, b) => b.score - a.score);
        });
        
        return rankings;
    }

    // Generar recomendaciones basadas en el análisis
    generateRecommendations() {
        const metrics = this.calculateAdvancedMetrics();
        const rankings = this.generateModelRanking();
        const recommendations = [];
        
        // Recomendación para producción
        const bestOverall = rankings.overall[0];
        recommendations.push({
            type: 'production',
            title: 'Mejor para Producción',
            model: bestOverall.model,
            reason: `Mejor balance general con score de ${bestOverall.score.toFixed(3)}`,
            icon: '🏆'
        });
        
        // Recomendación para accuracy
        const bestAccuracy = rankings.accuracy[0];
        if (bestAccuracy.key !== bestOverall.key) {
            recommendations.push({
                type: 'accuracy',
                title: 'Mejor Precisión',
                model: bestAccuracy.model,
                reason: `Máxima accuracy con ${bestAccuracy.score.toFixed(1)}%`,
                icon: '🎯'
            });
        }
        
        // Recomendación para velocidad
        const bestSpeed = rankings.speed[0];
        if (bestSpeed.key !== bestOverall.key) {
            recommendations.push({
                type: 'speed',
                title: 'Mejor Velocidad',
                model: bestSpeed.model,
                reason: `Máxima velocidad con ${bestSpeed.score.toFixed(1)} FPS`,
                icon: '⚡'
            });
        }
        
        // Análisis de clases problemáticas
        const problematicClasses = this.findProblematicClasses();
        if (problematicClasses.length > 0) {
            recommendations.push({
                type: 'warning',
                title: 'Clases Problemáticas',
                model: 'Todos los modelos',
                reason: `Clases con baja performance: ${problematicClasses.join(', ')}`,
                icon: '⚠️'
            });
        }
        
        return recommendations;
    }

    findProblematicClasses() {
        const classPerformance = {};
        
        // Calcular performance promedio por clase
        this.classNames.forEach(className => {
            const accuracies = [];
            Object.keys(this.modelData).forEach(key => {
                const classMetric = this.modelData[key].performance_metrics.per_class_metrics[className];
                if (classMetric && classMetric.accuracy > 0) {
                    accuracies.push(classMetric.accuracy);
                }
            });
            
            if (accuracies.length > 0) {
                classPerformance[className] = accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length;
            }
        });
        
        // Identificar clases con performance < 50%
        return Object.keys(classPerformance).filter(className => classPerformance[className] < 50);
    }

    // Exportar análisis completo
    exportAnalysis() {
        const analysis = {
            timestamp: new Date().toISOString(),
            models_analyzed: Object.keys(this.modelData).length,
            advanced_metrics: this.calculateAdvancedMetrics(),
            rankings: this.generateModelRanking(),
            recommendations: this.generateRecommendations(),
            problematic_classes: this.findProblematicClasses()
        };
        
        const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `model_analysis_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        window.URL.revokeObjectURL(url);
        
        return analysis;
    }
}

// Instancia global
window.modelAnalytics = new ModelAnalytics();