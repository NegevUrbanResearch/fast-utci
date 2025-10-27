/**
 * Time Controller for Full Day Analysis
 * 
 * Manages time slider and hour selection for animated UTCI visualization.
 */

/**
 * Create time control UI
 * @param {number} numHours - Number of hours in analysis
 * @param {function} onHourChange - Callback function(hourIndex) called when hour changes
 * @returns {HTMLElement} Control panel element
 */
export function createTimeControls(numHours, onHourChange) {
    const container = document.createElement('div');
    container.id = 'time-controls';
    container.style.cssText = `
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(255, 255, 255, 0.95);
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        z-index: 100;
        min-width: 400px;
    `;
    
    // Title
    const title = document.createElement('div');
    title.textContent = 'Time of Day';
    title.style.cssText = `
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
        font-size: 14px;
    `;
    container.appendChild(title);
    
    // Hour display
    const hourDisplay = document.createElement('div');
    hourDisplay.id = 'hour-display';
    hourDisplay.style.cssText = `
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    `;
    hourDisplay.textContent = '00:00';
    container.appendChild(hourDisplay);
    
    // Slider container
    const sliderContainer = document.createElement('div');
    sliderContainer.style.cssText = `
        display: flex;
        align-items: center;
        gap: 10px;
    `;
    
    // Slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = 'hour-slider';
    slider.min = '0';
    slider.max = String(numHours - 1);
    slider.value = '0';
    slider.style.cssText = `
        flex: 1;
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #313695, #4575B4, #74ADD1, #ABD9E9, #E0F3F8, #FFFFBF, #FEE090, #FDAE61, #F46D43, #D73027, #A50026);
        outline: none;
        cursor: pointer;
    `;
    
    // Slider input event
    slider.addEventListener('input', (e) => {
        const hourIndex = parseInt(e.target.value);
        updateHourDisplay(hourDisplay, hourIndex);
        onHourChange(hourIndex);
    });
    
    // Initialize display with current slider value
    updateHourDisplay(hourDisplay, 0);
    
    sliderContainer.appendChild(slider);
    container.appendChild(sliderContainer);
    
    // Play/Pause controls
    const controlButtons = document.createElement('div');
    controlButtons.style.cssText = `
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 10px;
    `;
    
    // Play button
    const playButton = document.createElement('button');
    playButton.textContent = '▶ Play';
    playButton.style.cssText = `
        padding: 8px 16px;
        border: none;
        border-radius: 5px;
        background: #4CAF50;
        color: white;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
    `;
    playButton.addEventListener('click', () => {
        if (playButton.dataset.playing === 'true') {
            stopAnimation();
            playButton.textContent = '▶ Play';
            playButton.dataset.playing = 'false';
        } else {
            startAnimation(slider, numHours, onHourChange);
            playButton.textContent = '⏸ Pause';
            playButton.dataset.playing = 'true';
        }
    });
    playButton.dataset.playing = 'false';
    
    controlButtons.appendChild(playButton);
    container.appendChild(controlButtons);
    
    return container;
}

/**
 * Update hour display text
 * @param {HTMLElement} display - Hour display element
 * @param {number} hourIndex - Hour index (0-23)
 */
function updateHourDisplay(display, hourIndex) {
    const hour = hourIndex;
    display.textContent = `${hour.toString().padStart(2, '0')}:00`;
}

// Animation state
let animationInterval = null;

/**
 * Start time animation
 * @param {HTMLInputElement} slider - Hour slider element
 * @param {number} numHours - Total number of hours
 * @param {function} onHourChange - Callback function
 */
function startAnimation(slider, numHours, onHourChange) {
    if (animationInterval) {
        clearInterval(animationInterval);
    }
    
    animationInterval = setInterval(() => {
        let currentHour = parseInt(slider.value);
        currentHour = (currentHour + 1) % numHours;
        slider.value = String(currentHour);
        
        // Trigger input event
        slider.dispatchEvent(new Event('input'));
    }, 500); // Change hour every 500ms
}

/**
 * Stop time animation
 */
function stopAnimation() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

/**
 * Set current hour programmatically
 * @param {number} hourIndex - Hour index to set
 */
export function setHour(hourIndex) {
    const slider = document.getElementById('hour-slider');
    const hourDisplay = document.getElementById('hour-display');
    
    if (slider && hourDisplay) {
        slider.value = String(hourIndex);
        updateHourDisplay(hourDisplay, hourIndex);
    }
}

/**
 * Get current hour
 * @returns {number} Current hour index
 */
export function getCurrentHour() {
    const slider = document.getElementById('hour-slider');
    return slider ? parseInt(slider.value) : 0;
}
