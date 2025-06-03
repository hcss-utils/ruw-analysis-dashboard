// Custom loading script for smoother component loading and radar pulse effect
window.addEventListener('DOMContentLoaded', function() {
    // Enhanced radar pulse loading animation
    // Create a MutationObserver to watch for loading elements
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    // Check if it's a loading element
                    if (node.classList && (
                        node.classList.contains('_dash-loading') || 
                        node.classList.contains('_dash-loading-callback') ||
                        node.classList.contains('dash-spinner')
                    )) {
                        enhanceLoadingSpinner(node);
                    }
                    
                    // Also check children
                    const loadingElements = node.querySelectorAll('._dash-loading, ._dash-loading-callback, .dash-spinner');
                    loadingElements.forEach(enhanceLoadingSpinner);
                }
            });
        });
    });

    // Start observing the document body for changes
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    function enhanceLoadingSpinner(element) {
        // Skip if already enhanced
        if (element.dataset.enhanced === 'true') return;
        
        // Mark as enhanced
        element.dataset.enhanced = 'true';
        
        // Hide any existing spinner content
        const existingContent = element.querySelectorAll('svg, div');
        existingContent.forEach(el => {
            el.style.display = 'none';
        });
        
        // Create radar pulse structure
        const radarContainer = document.createElement('div');
        radarContainer.className = 'radar-pulse-container';
        radarContainer.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100px;
            height: 100px;
            pointer-events: none;
        `;
        
        // Create multiple pulse rings
        for (let i = 0; i < 3; i++) {
            const pulse = document.createElement('div');
            pulse.className = 'radar-pulse-ring';
            pulse.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 30px;
                height: 30px;
                border: 3px solid rgba(19, 55, 111, ${1 - i * 0.3});
                border-radius: 50%;
                animation: radarPulse 2s infinite ease-out;
                animation-delay: ${i * 0.5}s;
                pointer-events: none;
            `;
            radarContainer.appendChild(pulse);
        }
        
        // Create center dot
        const centerDot = document.createElement('div');
        centerDot.className = 'radar-center-dot';
        centerDot.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 12px;
            height: 12px;
            background-color: #13376f;
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(19, 55, 111, 0.6);
            pointer-events: none;
        `;
        radarContainer.appendChild(centerDot);
        
        // Add radar to element
        element.appendChild(radarContainer);
        
        // Ensure the element is visible and positioned correctly
        element.style.position = 'fixed';
        element.style.top = '50%';
        element.style.left = '50%';
        element.style.transform = 'translate(-50%, -50%)';
        element.style.width = '120px';
        element.style.height = '120px';
        element.style.zIndex = '9999';
        element.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
        element.style.borderRadius = '10px';
        element.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.15)';
        element.style.display = 'flex';
        element.style.alignItems = 'center';
        element.style.justifyContent = 'center';
    }
    
    // Add the animation if not already present
    if (!document.querySelector('#radar-pulse-animation')) {
        const style = document.createElement('style');
        style.id = 'radar-pulse-animation';
        style.textContent = `
            @keyframes radarPulse {
                0% {
                    width: 30px;
                    height: 30px;
                    opacity: 1;
                    border-width: 3px;
                }
                100% {
                    width: 100px;
                    height: 100px;
                    opacity: 0;
                    border-width: 1px;
                }
            }
            
            /* Ensure Dash loading elements are visible when active */
            ._dash-loading, ._dash-loading-callback {
                display: flex !important;
                visibility: visible !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Create custom loading indicator
    var loadingDiv = document.createElement('div');
    loadingDiv.id = 'custom-loading-indicator';
    loadingDiv.style.position = 'fixed';
    loadingDiv.style.top = '50%';
    loadingDiv.style.left = '50%';
    loadingDiv.style.transform = 'translate(-50%, -50%)';
    loadingDiv.style.backgroundColor = '#ffffff';
    loadingDiv.style.padding = '20px';
    loadingDiv.style.borderRadius = '5px';
    loadingDiv.style.boxShadow = '0 0 10px rgba(0,0,0,0.2)';
    loadingDiv.style.zIndex = '10000';
    loadingDiv.style.display = 'none';
    
    // Loading text
    var loadingText = document.createElement('p');
    loadingText.textContent = 'Loading application components...';
    loadingText.style.margin = '0 0 10px 0';
    loadingText.style.fontWeight = 'bold';
    
    // Create radar pulse for custom loading
    var radarDiv = document.createElement('div');
    radarDiv.style.position = 'relative';
    radarDiv.style.width = '100px';
    radarDiv.style.height = '100px';
    radarDiv.style.margin = '0 auto';
    
    // Create radar rings
    for (let i = 0; i < 3; i++) {
        const ring = document.createElement('div');
        ring.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 30px;
            height: 30px;
            border: 3px solid rgba(19, 55, 111, ${1 - i * 0.3});
            border-radius: 50%;
            animation: radarPulse 2s infinite ease-out;
            animation-delay: ${i * 0.5}s;
        `;
        radarDiv.appendChild(ring);
    }
    
    // Add center dot
    const dot = document.createElement('div');
    dot.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 12px;
        height: 12px;
        background-color: #13376f;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(19, 55, 111, 0.6);
    `;
    radarDiv.appendChild(dot);
    
    // Assemble loading indicator
    loadingDiv.appendChild(loadingText);
    loadingDiv.appendChild(radarDiv);
    document.body.appendChild(loadingDiv);
    
    // Show loading indicator on component loading errors
    window.addEventListener('error', function(event) {
        if (event.filename && event.filename.includes('dash') && event.filename.includes('async')) {
            console.log('Caught component loading error, showing loading indicator');
            loadingDiv.style.display = 'block';
            
            // Try to refresh the page after a short delay
            setTimeout(function() {
                window.location.reload();
            }, 3000);
            
            // Prevent default error handling
            event.preventDefault();
        }
    }, true);
    
    // Hide loading indicator when app is fully loaded
    window.addEventListener('load', function() {
        loadingDiv.style.display = 'none';
    });
});