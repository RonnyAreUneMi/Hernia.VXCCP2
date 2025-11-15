// Dashboard Charts - Funciones auxiliares para gráficas avanzadas
class DashboardCharts {
    constructor() {
        this.modelData = {};
        this.modelColors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4'];
        this.modelColorMap = {
            'retinanet': '#3b82f6',
            'yolo': '#ef4444', 
            'resnet': '#10b981',
            'efficientnet': '#f59e0b',
            'mobilenet': '#8b5cf6',
            'densenet': '#06b6d4'
        };
        this.classNames = ['Hernia', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'Sin Hernia', 'T12'];
    }

    // Configuración base para gráficas
    getBaseChartConfig(isDark = false) {
        return {
            chart: {
                background: 'transparent',
                foreColor: isDark ? '#cbd5e1' : '#475569',
                fontFamily: 'Inter, system-ui, sans-serif',
                toolbar: {
                    show: true,
                    tools: {
                        download: true,
                        selection: true,
                        zoom: true,
                        zoomin: true,
                        zoomout: true,
                        pan: true,
                        reset: true
                    }
                }
            },
            tooltip: {
                theme: isDark ? 'dark' : 'light',
                style: {
                    fontSize: '13px',
                    fontFamily: 'Inter, system-ui, sans-serif'
                }
            },
            legend: {
                labels: {
                    colors: isDark ? '#cbd5e1' : '#475569'
                }
            },
            grid: {
                borderColor: isDark ? '#374151' : '#e5e7eb',
                strokeDashArray: 3
            },
            xaxis: {
                labels: {
                    style: {
                        colors: isDark ? '#94a3b8' : '#64748b'
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: isDark ? '#94a3b8' : '#64748b'
                    }
                }
            }
        };
    }

    // Gráfica de comparación de loss combinada
    drawCombinedLossChart(containerId) {
        const series = [];
        const isDark = document.documentElement.classList.contains('dark');
        
        Object.keys(this.modelData).forEach(key => {
            const data = this.modelData[key];
            const trainHistory = data.training_results.train_loss_history;
            const valHistory = data.training_results.val_loss_history;
            
            // Training loss
            series.push({
                name: `${data.model_info.architecture} - Training`,
                data: trainHistory.map((loss, epoch) => ({ x: epoch + 1, y: parseFloat(loss.toFixed(4)) })),
                color: this.modelColorMap[key] || this.modelColors[0],
                type: 'line'
            });
            
            // Validation loss
            series.push({
                name: `${data.model_info.architecture} - Validation`,
                data: valHistory.map((loss, epoch) => ({ x: epoch + 1, y: parseFloat(loss.toFixed(4)) })),
                color: this.modelColorMap[key] || this.modelColors[0],
                type: 'line',
                stroke: {
                    dashArray: 5
                }
            });
        });

        const options = {
            ...this.getBaseChartConfig(isDark),
            series: series,
            chart: {
                ...this.getBaseChartConfig(isDark).chart,
                type: 'line',
                height: 450
            },
            stroke: {
                curve: 'smooth',
                width: 3
            },
            markers: {
                size: 4
            },
            xaxis: {
                ...this.getBaseChartConfig(isDark).xaxis,
                title: {
                    text: 'Épocas',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            },
            yaxis: {
                ...this.getBaseChartConfig(isDark).yaxis,
                title: {
                    text: 'Loss',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            }
        };

        const chart = new ApexCharts(document.getElementById(containerId), options);
        chart.render();
    }

    // Gráfica de métricas por época (si disponible)
    drawMetricsEvolutionChart(containerId) {
        if (!this.hasEpochMetrics()) return;
        
        const series = [];
        const isDark = document.documentElement.classList.contains('dark');
        
        Object.keys(this.modelData).forEach(key => {
            const data = this.modelData[key];
            // Aquí podrías agregar métricas por época si están disponibles
            // Por ahora, usaremos los datos de loss como ejemplo
        });

        // Implementar según datos disponibles
    }

    // Gráfica de distribución de clases
    drawClassDistributionChart(containerId) {
        const series = [];
        const isDark = document.documentElement.classList.contains('dark');
        
        Object.keys(this.modelData).forEach(key => {
            const data = this.modelData[key];
            const classCounts = this.classNames.map(className => {
                const classMetrics = data.performance_metrics.per_class_metrics[className];
                return classMetrics ? classMetrics.total_samples : 0;
            });
            
            series.push({
                name: data.model_info.architecture || data.model_info.name,
                data: classCounts,
                color: this.modelColorMap[key] || this.modelColors[0]
            });
        });

        const options = {
            ...this.getBaseChartConfig(isDark),
            series: series,
            chart: {
                ...this.getBaseChartConfig(isDark).chart,
                type: 'bar',
                height: 400,
                stacked: true
            },
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: '70%'
                }
            },
            xaxis: {
                ...this.getBaseChartConfig(isDark).xaxis,
                categories: this.classNames,
                title: {
                    text: 'Clases',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            },
            yaxis: {
                ...this.getBaseChartConfig(isDark).yaxis,
                title: {
                    text: 'Número de Muestras',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            }
        };

        const chart = new ApexCharts(document.getElementById(containerId), options);
        chart.render();
    }

    // Gráfica de tiempo de entrenamiento vs accuracy
    drawTrainingEfficiencyChart(containerId) {
        const series = [{
            name: 'Modelos',
            data: Object.keys(this.modelData).map(key => {
                const data = this.modelData[key];
                return {
                    x: data.speed_metrics.training_time_minutes || 0,
                    y: data.performance_metrics.overall_accuracy,
                    z: data.model_info.architecture || data.model_info.name,
                    fillColor: this.modelColorMap[key] || this.modelColors[0]
                };
            })
        }];

        const isDark = document.documentElement.classList.contains('dark');

        const options = {
            ...this.getBaseChartConfig(isDark),
            series: series,
            chart: {
                ...this.getBaseChartConfig(isDark).chart,
                type: 'scatter',
                height: 400
            },
            xaxis: {
                ...this.getBaseChartConfig(isDark).xaxis,
                title: {
                    text: 'Tiempo de Entrenamiento (minutos)',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            },
            yaxis: {
                ...this.getBaseChartConfig(isDark).yaxis,
                title: {
                    text: 'Accuracy (%)',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            },
            tooltip: {
                ...this.getBaseChartConfig(isDark).tooltip,
                custom: function({series, seriesIndex, dataPointIndex, w}) {
                    const point = w.globals.initialSeries[seriesIndex].data[dataPointIndex];
                    return `<div class="p-3">
                        <strong>${point.z}</strong><br/>
                        Tiempo: ${point.x} min<br/>
                        Accuracy: ${point.y.toFixed(1)}%
                    </div>`;
                }
            }
        };

        const chart = new ApexCharts(document.getElementById(containerId), options);
        chart.render();
    }

    // Análisis de convergencia
    drawConvergenceAnalysis(containerId) {
        const series = [];
        const isDark = document.documentElement.classList.contains('dark');
        
        Object.keys(this.modelData).forEach(key => {
            const data = this.modelData[key];
            const valHistory = data.training_results.val_loss_history;
            
            // Calcular la mejora por época
            const improvements = [];
            let bestLoss = Infinity;
            
            valHistory.forEach((loss, epoch) => {
                if (loss < bestLoss) {
                    bestLoss = loss;
                    improvements.push({ x: epoch + 1, y: 1 }); // Mejora
                } else {
                    improvements.push({ x: epoch + 1, y: 0 }); // Sin mejora
                }
            });
            
            series.push({
                name: data.model_info.architecture || data.model_info.name,
                data: improvements,
                color: this.modelColorMap[key] || this.modelColors[0]
            });
        });

        const options = {
            ...this.getBaseChartConfig(isDark),
            series: series,
            chart: {
                ...this.getBaseChartConfig(isDark).chart,
                type: 'area',
                height: 350,
                stacked: false
            },
            fill: {
                opacity: 0.6
            },
            xaxis: {
                ...this.getBaseChartConfig(isDark).xaxis,
                title: {
                    text: 'Épocas',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            },
            yaxis: {
                ...this.getBaseChartConfig(isDark).yaxis,
                title: {
                    text: 'Mejora (1=Sí, 0=No)',
                    style: {
                        color: isDark ? '#cbd5e1' : '#475569'
                    }
                }
            }
        };

        const chart = new ApexCharts(document.getElementById(containerId), options);
        chart.render();
    }

    // Utilidades
    hasEpochMetrics() {
        return Object.values(this.modelData).some(data => 
            data.training_results && data.training_results.train_loss_history
        );
    }

    setModelData(data) {
        this.modelData = data;
    }

    // Exportar datos a CSV
    exportToCSV(filename = 'model_comparison.csv') {
        const headers = ['Modelo', 'Accuracy', 'FPS', 'Best_Val_Loss', 'Epochs', 'Training_Time'];
        const rows = [headers];
        
        Object.keys(this.modelData).forEach(key => {
            const data = this.modelData[key];
            rows.push([
                data.model_info.architecture || data.model_info.name,
                data.performance_metrics.overall_accuracy.toFixed(2),
                data.speed_metrics.fps.toFixed(2),
                data.training_results.best_val_loss.toFixed(4),
                data.training_config.epochs_trained || data.training_config.epochs_configured,
                data.speed_metrics.training_time_minutes || 'N/A'
            ]);
        });
        
        const csvContent = rows.map(row => row.join(',')).join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }
}

// Instancia global
window.dashboardCharts = new DashboardCharts();