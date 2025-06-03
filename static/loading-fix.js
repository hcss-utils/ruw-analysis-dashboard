// Enhanced loading script with proper cleanup logic
window.addEventListener('DOMContentLoaded', function() {
    // Track all active loading states
    const activeLoadingStates = new Set();
    
    // Enhanced radar pulse loading animation
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            // Check for added nodes
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    checkForLoadingElements(node);
                }
            });
            
            // Check for removed nodes
            mutation.removedNodes.forEach(function(node) {
                if (node.nodeType === 1) {
                    checkForRemovedLoadingElements(node);
                }
            });
            
            // Check for attribute changes
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-dash-is-loading') {
                if (mutation.target.getAttribute('data-dash-is-loading') === 'true') {
                    activeLoadingStates.add(mutation.target);
                    showRadarSweep();
                } else {
                    activeLoadingStates.delete(mutation.target);
                    checkIfShouldHideRadarSweep();
                }
            }
        });
    });
    
    function checkForLoadingElements(node) {
        // Check if it's a loading element
        if (node.classList && (
            node.classList.contains('_dash-loading') || 
            node.classList.contains('_dash-loading-callback') ||
            node.classList.contains('dash-spinner')
        )) {
            activeLoadingStates.add(node);
            enhanceLoadingSpinner(node);
        }
        
        // Check if it has loading attribute
        if (node.getAttribute && node.getAttribute('data-dash-is-loading') === 'true') {
            activeLoadingStates.add(node);
            showRadarSweep();
        }
        
        // Also check children
        const loadingElements = node.querySelectorAll('._dash-loading, ._dash-loading-callback, .dash-spinner, [data-dash-is-loading="true"]');
        loadingElements.forEach(function(el) {
            activeLoadingStates.add(el);
            enhanceLoadingSpinner(el);
        });
    }
    
    function checkForRemovedLoadingElements(node) {
        activeLoadingStates.delete(node);
        
        // Also check children that might have been removed
        const loadingElements = node.querySelectorAll('._dash-loading, ._dash-loading-callback, .dash-spinner, [data-dash-is-loading="true"]');
        loadingElements.forEach(function(el) {
            activeLoadingStates.delete(el);
        });
        
        checkIfShouldHideRadarSweep();
    }
    
    function checkIfShouldHideRadarSweep() {
        // Only hide radar sweep if NO loading elements remain
        if (activeLoadingStates.size === 0) {
            // Double-check DOM for any loading elements we might have missed
            const stillLoading = document.querySelector('._dash-loading:not(:empty), ._dash-loading-callback:not(:empty), .dash-spinner:not(:empty), [data-dash-is-loading="true"]');
            if (!stillLoading) {
                hideRadarSweep();
            }
        }
    }

    // Start observing the document body for changes
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-dash-is-loading']
    });

    function enhanceLoadingSpinner(element) {
        // Skip if already enhanced
        if (element.dataset.enhanced === 'true') return;
        
        // Mark as enhanced
        element.dataset.enhanced = 'true';
        
        // Hide the default loading element
        element.style.display = 'none';
        element.style.visibility = 'hidden';
        
        showRadarSweep();
    }
    
    function showRadarSweep() {
        // Create horizontal radar sweep if not already present
        if (!document.querySelector('.radar-sweep-container')) {
            // Create radar sweep container
            const sweepContainer = document.createElement('div');
            sweepContainer.className = 'radar-sweep-container';
            sweepContainer.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 9999;
                overflow: hidden;
            `;
            
            // Create the sweep band
            const sweepBand = document.createElement('div');
            sweepBand.className = 'radar-sweep-band';
            sweepBand.style.cssText = `
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    to right,
                    transparent 0%,
                    rgba(19, 55, 111, 0.05) 20%,
                    rgba(19, 55, 111, 0.1) 40%,
                    rgba(19, 55, 111, 0.15) 50%,
                    rgba(19, 55, 111, 0.1) 60%,
                    rgba(19, 55, 111, 0.05) 80%,
                    transparent 100%
                );
                animation: horizontalRadarSweep 2s ease-in-out infinite;
            `;
            
            // Create subtle background overlay
            const overlay = document.createElement('div');
            overlay.className = 'radar-sweep-overlay';
            overlay.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(255, 255, 255, 0.3);
                pointer-events: none;
            `;
            
            sweepContainer.appendChild(overlay);
            sweepContainer.appendChild(sweepBand);
            document.body.appendChild(sweepContainer);
            
            // Clear any existing timeout
            if (window.radarSweepTimeout) {
                clearTimeout(window.radarSweepTimeout);
            }
            
            // Set a longer timeout as fallback
            window.radarSweepTimeout = setTimeout(() => {
                hideRadarSweep();
            }, 30000); // 30 seconds max
        }
    }
    
    function hideRadarSweep() {
        const sweepContainer = document.querySelector('.radar-sweep-container');
        if (sweepContainer) {
            sweepContainer.remove();
        }
        
        // Clear timeout if it exists
        if (window.radarSweepTimeout) {
            clearTimeout(window.radarSweepTimeout);
            window.radarSweepTimeout = null;
        }
        
        // Clear active loading states
        activeLoadingStates.clear();
    }
    
    // Add the animation if not already present
    if (!document.querySelector('#radar-sweep-animation')) {
        const style = document.createElement('style');
        style.id = 'radar-sweep-animation';
        style.textContent = `
            @keyframes horizontalRadarSweep {
                0% {
                    left: -100%;
                }
                100% {
                    left: 100%;
                }
            }
            
            /* Ensure radar sweep is visible */
            .radar-sweep-container {
                display: block !important;
                visibility: visible !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Create custom loading indicator for errors only
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
    loadingText.style.margin = '0';
    loadingText.style.fontWeight = 'bold';
    
    loadingDiv.appendChild(loadingText);
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