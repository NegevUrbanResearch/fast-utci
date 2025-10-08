/**
 * UTCI Data Loader for Binary Format
 * 
 * Loads and parses binary UTCI data files exported by export_for_viewer.py
 * Supports both single hour and full day analysis formats.
 */

/**
 * Load analysis metadata from JSON file
 * @param {string} metadataPath - Path to metadata JSON file
 * @returns {Promise<object>} Metadata object
 */
export async function loadMetadata(metadataPath) {
    const response = await fetch(metadataPath);
    if (!response.ok) {
        throw new Error(`Failed to load metadata: ${response.statusText}`);
    }
    return await response.json();
}

/**
 * Parse single hour binary data
 * 
 * Binary structure:
 *   [4 bytes: num_positions (uint32)]
 *   [num_positions × 12 bytes: positions as float32 x,y,z]
 *   [num_positions × 4 bytes: utci values as float32]
 * 
 * @param {ArrayBuffer} buffer - Binary data buffer
 * @returns {object} Parsed data with positions and utci arrays
 */
function parseSingleHourBinary(buffer) {
    const dataView = new DataView(buffer);
    let offset = 0;
    
    // Read header
    const numPositions = dataView.getUint32(offset, true); // little-endian
    offset += 4;
    
    // Read positions (3 floats per position)
    const positions = new Float32Array(numPositions * 3);
    for (let i = 0; i < numPositions * 3; i++) {
        positions[i] = dataView.getFloat32(offset, true);
        offset += 4;
    }
    
    // Read UTCI values (1 float per position)
    const utciValues = new Float32Array(numPositions);
    for (let i = 0; i < numPositions; i++) {
        utciValues[i] = dataView.getFloat32(offset, true);
        offset += 4;
    }
    
    return {
        numPositions,
        positions,
        utciValues,
        numHours: 1
    };
}

/**
 * Parse full day binary data
 * 
 * Binary structure:
 *   [8 bytes: num_positions (uint32), num_hours (uint32)]
 *   [num_positions × 12 bytes: positions as float32 x,y,z]
 *   [num_positions × 4 bytes: utci values hour 0 as float32]
 *   [num_positions × 4 bytes: utci values hour 1 as float32]
 *   ...
 *   [num_positions × 4 bytes: utci values hour 23 as float32]
 * 
 * @param {ArrayBuffer} buffer - Binary data buffer
 * @returns {object} Parsed data with positions and utci arrays organized by hour
 */
function parseFullDayBinary(buffer) {
    const dataView = new DataView(buffer);
    let offset = 0;
    
    // Read header
    const numPositions = dataView.getUint32(offset, true);
    offset += 4;
    const numHours = dataView.getUint32(offset, true);
    offset += 4;
    
    // Read positions (3 floats per position)
    const positions = new Float32Array(numPositions * 3);
    for (let i = 0; i < numPositions * 3; i++) {
        positions[i] = dataView.getFloat32(offset, true);
        offset += 4;
    }
    
    // Read UTCI values hour by hour
    const utciByHour = [];
    for (let hour = 0; hour < numHours; hour++) {
        const hourValues = new Float32Array(numPositions);
        for (let i = 0; i < numPositions; i++) {
            hourValues[i] = dataView.getFloat32(offset, true);
            offset += 4;
        }
        utciByHour.push(hourValues);
    }
    
    return {
        numPositions,
        numHours,
        positions,
        utciByHour  // Array of Float32Arrays, one per hour
    };
}

/**
 * Load binary UTCI data file
 * @param {string} binaryPath - Path to binary data file
 * @param {string} analysisType - "single_hour" or "full_day"
 * @returns {Promise<object>} Parsed data object
 */
export async function loadBinaryData(binaryPath, analysisType) {
    const response = await fetch(binaryPath);
    if (!response.ok) {
        throw new Error(`Failed to load binary data: ${response.statusText}`);
    }
    
    const buffer = await response.arrayBuffer();
    
    if (analysisType === 'single_hour') {
        return parseSingleHourBinary(buffer);
    } else {
        return parseFullDayBinary(buffer);
    }
}

/**
 * Load complete UTCI analysis (metadata + binary data)
 * @param {string} analysisId - Analysis identifier (e.g., "20250815_grid_2m_fullday")
 * @param {string} dataDir - Base directory for data files (default: "../data/analyses")
 * @returns {Promise<object>} Complete analysis data with metadata and binary data
 */
export async function loadAnalysis(analysisId, dataDir = '../data/analyses') {
    const metadataPath = `${dataDir}/${analysisId}.json`;
    const binaryPath = `${dataDir}/${analysisId}.bin`;
    
    console.log(`[LOAD] Loading analysis: ${analysisId}`);
    
    // Load metadata
    const metadata = await loadMetadata(metadataPath);
    console.log(`[OK] Metadata loaded: ${metadata.num_positions} positions, ${metadata.hours.length} hours`);
    
    // Load binary data
    const binaryData = await loadBinaryData(binaryPath, metadata.analysis_type);
    console.log(`[OK] Binary data loaded: ${(binaryData.positions.length * 4 / 1024).toFixed(1)} KB`);
    
    return {
        metadata,
        data: binaryData
    };
}

/**
 * Get UTCI value for a specific position and hour
 * @param {object} analysis - Analysis data from loadAnalysis()
 * @param {number} positionIndex - Position index (0 to numPositions-1)
 * @param {number} hourIndex - Hour index (0 to numHours-1)
 * @returns {number} UTCI value in Celsius
 */
export function getUTCIValue(analysis, positionIndex, hourIndex = 0) {
    const { data } = analysis;
    
    if (data.numHours === 1) {
        // Single hour
        return data.utciValues[positionIndex];
    } else {
        // Full day
        return data.utciByHour[hourIndex][positionIndex];
    }
}

/**
 * Get position coordinates for a specific position
 * @param {object} analysis - Analysis data from loadAnalysis()
 * @param {number} positionIndex - Position index (0 to numPositions-1)
 * @returns {object} Position {x, y, z}
 */
export function getPosition(analysis, positionIndex) {
    const { data } = analysis;
    const i = positionIndex * 3;
    return {
        x: data.positions[i],
        y: data.positions[i + 1],
        z: data.positions[i + 2]
    };
}

/**
 * Get all UTCI values for a specific hour
 * @param {object} analysis - Analysis data from loadAnalysis()
 * @param {number} hourIndex - Hour index (0 to numHours-1)
 * @returns {Float32Array} Array of UTCI values
 */
export function getUTCIForHour(analysis, hourIndex = 0) {
    const { data } = analysis;
    
    if (data.numHours === 1) {
        return data.utciValues;
    } else {
        return data.utciByHour[hourIndex];
    }
}

/**
 * Calculate statistics for UTCI data
 * @param {Float32Array} utciValues - Array of UTCI values
 * @returns {object} Statistics {min, max, mean}
 */
export function calculateStatistics(utciValues) {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let count = 0;
    
    for (let i = 0; i < utciValues.length; i++) {
        const val = utciValues[i];
        if (!isNaN(val) && isFinite(val)) {
            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
            count++;
        }
    }
    
    return {
        min,
        max,
        mean: count > 0 ? sum / count : 0,
        count
    };
}
