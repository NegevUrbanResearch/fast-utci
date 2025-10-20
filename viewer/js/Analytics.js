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
 * Calculate Pearson correlation coefficient
 * @param {Float32Array} values1 - First array of values
 * @param {Float32Array} values2 - Second array of values
 * @returns {number} Correlation coefficient (-1 to 1)
 */
function calculateCorrelation(values1, values2) {
    const minLength = Math.min(values1.length, values2.length);
    
    // Calculate means
    let sum1 = 0, sum2 = 0;
    let count = 0;
    
    for (let i = 0; i < minLength; i++) {
        if (!isNaN(values1[i]) && !isNaN(values2[i]) && 
            isFinite(values1[i]) && isFinite(values2[i])) {
            sum1 += values1[i];
            sum2 += values2[i];
            count++;
        }
    }
    
    if (count === 0) return 0;
    
    const mean1 = sum1 / count;
    const mean2 = sum2 / count;
    
    // Calculate correlation coefficient
    let numerator = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;
    
    for (let i = 0; i < minLength; i++) {
        if (!isNaN(values1[i]) && !isNaN(values2[i]) && 
            isFinite(values1[i]) && isFinite(values2[i])) {
            const diff1 = values1[i] - mean1;
            const diff2 = values2[i] - mean2;
            
            numerator += diff1 * diff2;
            sumSq1 += diff1 * diff1;
            sumSq2 += diff2 * diff2;
        }
    }
    
    const denominator = Math.sqrt(sumSq1 * sumSq2);
    return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Find spatially matched points between analysis and validation data
 * @param {object} analysis - Analysis data
 * @param {object} validation - Validation data
 * @returns {object} Matched analysis and validation values
 */
function findSpatiallyMatchedPoints(analysis, validation) {
    const analysisPositions = analysis.data.positions;
    const validationPositions = validation.positions;
    
    const matchedAnalysisValues = [];
    const matchedValidationValues = [];
    
    // For each validation point, find the closest analysis point
    for (let v = 0; v < validation.numPositions; v++) {
        const vx = validationPositions[v * 3];
        const vy = validationPositions[v * 3 + 1];
        const vz = validationPositions[v * 3 + 2];
        
        let closestDistance = Infinity;
        let closestIndex = -1;
        
        // Find closest analysis point
        for (let a = 0; a < analysis.data.numPositions; a++) {
            const ax = analysisPositions[a * 3];
            const ay = analysisPositions[a * 3 + 1];
            const az = analysisPositions[a * 3 + 2];
            
            const distance = Math.sqrt(
                (ax - vx) ** 2 + (ay - vy) ** 2 + (az - vz) ** 2
            );
            
            if (distance < closestDistance) {
                closestDistance = distance;
                closestIndex = a;
            }
        }
        
        if (closestIndex !== -1) {
            matchedAnalysisValues.push(closestIndex);
            matchedValidationValues.push(v);
        }
    }
    
    return { matchedAnalysisValues, matchedValidationValues };
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
    
    // Check if we need spatial matching (different number of points)
    let matchedAnalysisValues, matchedValidationValues;
    
    if (analysisValues.length !== validationValues.length) {
        console.log(`[COMPARISON] Spatial matching required: Analysis ${analysisValues.length} points vs Validation ${validationValues.length} points`);
        
        const spatialMatch = findSpatiallyMatchedPoints(analysis, validation);
        
        // Extract matched values
        matchedAnalysisValues = new Float32Array(spatialMatch.matchedAnalysisValues.length);
        matchedValidationValues = new Float32Array(spatialMatch.matchedValidationValues.length);
        
        for (let i = 0; i < spatialMatch.matchedAnalysisValues.length; i++) {
            matchedAnalysisValues[i] = analysisValues[spatialMatch.matchedAnalysisValues[i]];
            matchedValidationValues[i] = validationValues[spatialMatch.matchedValidationValues[i]];
        }
        
        console.log(`[COMPARISON] Spatial matching complete: ${matchedAnalysisValues.length} matched points`);
    } else {
        // Same number of points, use direct comparison
        matchedAnalysisValues = analysisValues;
        matchedValidationValues = validationValues;
    }
    
    // Calculate statistics for both datasets
    const analysisStats = calculateStatistics(analysisValues); // Use original for display
    const validationStats = calculateStatistics(validationValues); // Use original for display
    
    // Calculate statistics differences
    const statsDiff = calculateStatisticsDifferences(analysisStats, validationStats);
    
    // Calculate correlation using matched data
    const correlation = calculateCorrelation(matchedAnalysisValues, matchedValidationValues);
    
    return {
        analysis: analysisStats,
        validation: validationStats,
        comparison: {
            minDiff: statsDiff.minDiff,
            maxDiff: statsDiff.maxDiff,
            meanDiff: statsDiff.meanDiff,
            correlation
        }
    };
}

/**
 * Create analytics panel UI
 * @param {object} metadata - Analysis metadata
 * @param {object} comparisonStats - Statistics from compareWithValidation()
 * @returns {HTMLElement} Analytics panel element
 */
export function createAnalyticsPanel(metadata, comparisonStats = null) {
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
                Mean Diff: ${comparisonStats.comparison.meanDiff >= 0 ? '+' : ''}${comparisonStats.comparison.meanDiff.toFixed(2)}°C<br>
                Correlation: ${comparisonStats.comparison.correlation.toFixed(3)}
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
            metricsSection.innerHTML = `
                <strong>Comparison Metrics:</strong><br>
                Min Diff: ${comparisonStats.comparison.minDiff >= 0 ? '+' : ''}${comparisonStats.comparison.minDiff.toFixed(2)}°C<br>
                Max Diff: ${comparisonStats.comparison.maxDiff >= 0 ? '+' : ''}${comparisonStats.comparison.maxDiff.toFixed(2)}°C<br>
                Mean Diff: ${comparisonStats.comparison.meanDiff >= 0 ? '+' : ''}${comparisonStats.comparison.meanDiff.toFixed(2)}°C<br>
                Correlation: ${comparisonStats.comparison.correlation.toFixed(3)}
            `;
        }
    }
}
