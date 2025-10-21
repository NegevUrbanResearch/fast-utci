/**
 * Analytics Module for Grasshopper Validation Comparison
 * 
 * Compares UTCI results with Grasshopper validation data and displays statistics.
 */

import { loadBinaryData } from './UTCIDataLoader.js';
import { calculateStatistics } from './UTCIDataLoader.js';

/**
 * Load Grasshopper validation data
 * @param {string} validationPath - Path to validation binary file
 * @returns {Promise<object>} Validation data
 */
export async function loadValidationData(validationPath = '../data/validation/grasshopper_aug15_fullday.bin') {
    console.log('[LOAD] Loading Grasshopper validation data...');
    const data = await loadBinaryData(validationPath, 'full_day');
    console.log('[OK] Validation data loaded');
    return data;
}

/**
 * Calculate statistics differences between two datasets
 * @param {object} stats1 - Statistics object with min, max, mean
 * @param {object} stats2 - Statistics object with min, max, mean
 * @returns {object} Differences between statistics
 */
function calculateStatisticsDifferences(stats1, stats2) {
    return {
        minDiff: stats1.min - stats2.min,
        maxDiff: stats1.max - stats2.max,
        meanDiff: stats1.mean - stats2.mean
    };
}

/**
 * Calculate average mean difference across all 24 hours
 * @param {object} analysis - Analysis data from UTCIDataLoader
 * @param {object} validation - Validation data from loadValidationData()
 * @returns {number} Average mean difference across all hours
 */
export function calculateAvgMeanDiffAllHours(analysis, validation) {
    if (!analysis.data.utciByHour || analysis.data.numHours < 2) {
        return null;  // Not a full day analysis
    }
    
    let totalMeanDiff = 0;
    let validHours = 0;
    
    for (let h = 0; h < analysis.data.numHours; h++) {
        const analysisValues = analysis.data.utciByHour[h];
        const validationValues = validation.utciByHour[h];
        
        if (analysisValues && validationValues) {
            const analysisStats = calculateStatistics(analysisValues);
            const validationStats = calculateStatistics(validationValues);
            totalMeanDiff += analysisStats.mean - validationStats.mean;
            validHours++;
        }
    }
    
    return validHours > 0 ? totalMeanDiff / validHours : null;
}

/**
 * Compare analysis with validation data using spatial matching
 * @param {object} analysis - Analysis data from UTCIDataLoader
 * @param {object} validation - Validation data from loadValidationData()
 * @param {number} hourIndex - Hour to compare (default: 0)
 * @returns {object} Comparison statistics
 */
export function compareWithValidation(analysis, validation, hourIndex = 0) {
    // Get UTCI values for this hour
    let analysisValues;
    let validationHourIndex = hourIndex;
    
    if (analysis.data.numHours === 1) {
        analysisValues = analysis.data.utciValues;
        // For single hour analysis, use the specific hour from metadata
        validationHourIndex = analysis.metadata.hours[0]; // Get the actual hour (e.g., 13)
    } else {
        analysisValues = analysis.data.utciByHour[hourIndex];
        validationHourIndex = hourIndex;
    }
    
    const validationValues = validation.utciByHour[validationHourIndex];
    
    // Calculate statistics for both full datasets (no spatial matching needed)
    const analysisStats = calculateStatistics(analysisValues);
    const validationStats = calculateStatistics(validationValues);
    
    // Calculate statistics differences
    const statsDiff = calculateStatisticsDifferences(analysisStats, validationStats);
    
    return {
        analysis: analysisStats,
        validation: validationStats,
        comparison: {
            minDiff: statsDiff.minDiff,
            maxDiff: statsDiff.maxDiff,
            meanDiff: statsDiff.meanDiff
        }
    };
}

/**
 * Create analytics panel UI
 * @param {object} metadata - Analysis metadata
 * @param {object} comparisonStats - Statistics from compareWithValidation()
 * @param {number} avgMeanDiffAllHours - Average mean diff across all 24 hours (optional)
 * @returns {HTMLElement} Analytics panel element
 */
export function createAnalyticsPanel(metadata, comparisonStats = null, avgMeanDiffAllHours = null) {
    const panel = document.createElement('div');
    panel.id = 'analytics-panel';
    panel.style.cssText = `
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-width: 300px;
        z-index: 100;
    `;
    
    let content = `
        <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px; border-bottom: 2px solid #333; padding-bottom: 5px;">
            Analysis Info
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Date:</strong> ${metadata.date}<br>
            <strong>Grid Size:</strong> ${metadata.grid_size}m<br>
            <strong>Positions:</strong> ${metadata.num_positions.toLocaleString()}<br>
            <strong>Runtime:</strong> ${metadata.runtime_seconds.toFixed(1)}s
        </div>
        <div style="margin-bottom: 8px;">
            <strong>UTCI Range:</strong><br>
            Min: ${metadata.utci_range.min.toFixed(1)}°C<br>
            Max: ${metadata.utci_range.max.toFixed(1)}°C<br>
            Mean: ${metadata.utci_range.mean.toFixed(1)}°C
        </div>
    `;
    
    if (comparisonStats) {
        content += `
            <div style="font-weight: bold; font-size: 14px; margin: 15px 0 10px 0; border-bottom: 2px solid #333; padding-bottom: 5px;">
                Grasshopper Comparison
            </div>
            <div style="margin-bottom: 8px;">
                <strong>Validation Data:</strong><br>
                Min: ${comparisonStats.validation.min.toFixed(1)}°C<br>
                Max: ${comparisonStats.validation.max.toFixed(1)}°C<br>
                Mean: ${comparisonStats.validation.mean.toFixed(1)}°C
            </div>
            <div style="margin-bottom: 8px;">
                <strong>Comparison Metrics:</strong><br>
                Min Diff: ${comparisonStats.comparison.minDiff >= 0 ? '+' : ''}${comparisonStats.comparison.minDiff.toFixed(2)}°C<br>
                Max Diff: ${comparisonStats.comparison.maxDiff >= 0 ? '+' : ''}${comparisonStats.comparison.maxDiff.toFixed(2)}°C<br>
                Mean Diff: ${comparisonStats.comparison.meanDiff >= 0 ? '+' : ''}${comparisonStats.comparison.meanDiff.toFixed(2)}°C${avgMeanDiffAllHours !== null ? `<br>24-Hour Avg: ${avgMeanDiffAllHours >= 0 ? '+' : ''}${avgMeanDiffAllHours.toFixed(2)}°C` : ''}
            </div>
        `;
    }
    
    // Add sun path toggle for full day analysis with sun data
    if (metadata.analysis_type === 'full_day' && metadata.sun_positions) {
        content += `
            <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #ddd;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                    <input type="checkbox" id="sun-path-toggle" style="margin-right: 8px; cursor: pointer;">
                    <span style="font-size: 12px; font-weight: 500;">Show Sun Path</span>
                </label>
            </div>
        `;
    }
    
    panel.innerHTML = content;
    return panel;
}

/**
 * Update analytics panel with new hour data
 * @param {object} comparisonStats - New statistics from compareWithValidation()
 * @param {object} analysisStats - Current analysis statistics for this hour
 */
export function updateAnalyticsPanel(comparisonStats, analysisStats = null) {
    const panel = document.getElementById('analytics-panel');
    if (!panel) return;
    
    // Update UTCI range if analysis stats are provided (for full day analysis)
    if (analysisStats) {
        // Find the UTCI range section and update it
        const rangeDivs = panel.querySelectorAll('div');
        for (let i = 0; i < rangeDivs.length; i++) {
            const div = rangeDivs[i];
            if (div.textContent.includes('UTCI Range:')) {
                // Update the entire div content
                div.innerHTML = `
                    <strong>UTCI Range:</strong><br>
                    Min: ${analysisStats.min.toFixed(1)}°C<br>
                    Max: ${analysisStats.max.toFixed(1)}°C<br>
                    Mean: ${analysisStats.mean.toFixed(1)}°C
                `;
                break;
            }
        }
    }
    
    // Find the comparison section and update it
    const sections = panel.querySelectorAll('div');
    let comparisonSection = null;
    
    for (const section of sections) {
        if (section.textContent.includes('Grasshopper Comparison')) {
            comparisonSection = section.nextElementSibling;
            break;
        }
    }
    
    if (comparisonSection && comparisonStats) {
        comparisonSection.innerHTML = `
            <strong>Validation Data:</strong><br>
            Min: ${comparisonStats.validation.min.toFixed(1)}°C<br>
            Max: ${comparisonStats.validation.max.toFixed(1)}°C<br>
            Mean: ${comparisonStats.validation.mean.toFixed(1)}°C
        `;
        
        const metricsSection = comparisonSection.nextElementSibling;
        if (metricsSection) {
            // Preserve the 24-hour average line if it exists
            const existingContent = metricsSection.innerHTML;
            const avgMatch = existingContent.match(/24-Hour Avg: [+\-]?\d+\.\d+°C/);
            const avgLine = avgMatch ? `<br>${avgMatch[0]}` : '';
            
            metricsSection.innerHTML = `
                <strong>Comparison Metrics:</strong><br>
                Min Diff: ${comparisonStats.comparison.minDiff >= 0 ? '+' : ''}${comparisonStats.comparison.minDiff.toFixed(2)}°C<br>
                Max Diff: ${comparisonStats.comparison.maxDiff >= 0 ? '+' : ''}${comparisonStats.comparison.maxDiff.toFixed(2)}°C<br>
                Mean Diff: ${comparisonStats.comparison.meanDiff >= 0 ? '+' : ''}${comparisonStats.comparison.meanDiff.toFixed(2)}°C${avgLine}
            `;
        }
    }
}
