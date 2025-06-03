// Progressive loading controller for dashboard
(function() {
    'use strict';
    
    // Configuration
    const PRIORITY_COMPONENTS = {
        'explore': ['sunburst-chart', 'explore-chunks-container'],  // Sunburst AND chunks
        'search': ['search-results'],
        'compare': ['compare-viz-container'],
        'sources': ['sources-result-stats', 'sources-subtabs'],  // Monitor these for Sources tab
        'burstiness': ['burst-results-container']  // Add burstiness tab
    };
    
    // Track loading states
    const loadingStates = {
        priority: new Set(),
        background: new Set(),
        priorityComplete: false
    };
    
    // Helper to get current tab
    function getCurrentTab() {
        // Check which tab is active
        const activeTab = document.querySelector('.dash-tab--selected');
        if (activeTab) {
            const tabText = activeTab.textContent.toLowerCase();
            return tabText;
        }
        return 'explore'; // Default
    }
    
    // Override the loading monitor to distinguish priority vs background
    if (window.dashLoadingMonitor) {
        const originalGetStates = window.dashLoadingMonitor.getStates;
        const originalShow = window.dashLoadingMonitor.forceShow;
        const originalHide = window.dashLoadingMonitor.forceHide;
        
        // Track which components are priority
        window.dashLoadingMonitor.markPriority = function(componentId) {
            loadingStates.priority.add(componentId);
        };
        
        window.dashLoadingMonitor.markBackground = function(componentId) {
            loadingStates.background.add(componentId);
        };
        
        // Check if priority loading is complete
        window.dashLoadingMonitor.isPriorityComplete = function() {
            const states = originalGetStates();
            
            // Get current tab
            const currentTab = getCurrentTab();
            const priorityIds = PRIORITY_COMPONENTS[currentTab] || [];
            
            // Check if any priority components for current tab are still loading
            for (const id of priorityIds) {
                if (states.components.includes(id) || 
                    states.graphs.includes(id) ||
                    states.callbacks.includes(id)) {
                    return false;
                }
            }
            
            return true;
        };
    }
    
    // Monitor for priority component completion
    function setupProgressiveLoading() {
        const observer = new MutationObserver(function(mutations) {
            // Check if we need to reset loading state (new segment clicked)
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' && mutation.target.id === 'chunks-selection-title') {
                    // Title changed - new segment clicked
                    if (loadingStates.priorityComplete) {
                        console.log('[Progressive Loading] New segment clicked, resetting loading state');
                        loadingStates.priorityComplete = false;
                        // The radar pulse should already be showing from dash-loading-monitor
                    }
                }
            });
            
            checkPriorityCompletion();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['data-dash-is-loading']
        });
    }
    
    function checkPriorityCompletion() {
        if (!loadingStates.priorityComplete) {
            const currentTab = getCurrentTab();
            const priorityIds = PRIORITY_COMPONENTS[currentTab] || [];
            
            // Special handling for Sources tab
            if (currentTab === 'sources') {
                // Check if sources data is being loaded
                const sourcesStats = document.getElementById('sources-result-stats');
                const sourcesSubtabs = document.getElementById('sources-subtabs');
                
                // If stats div is empty or contains the "Click Apply Filters" message, no loading
                if (sourcesStats && (sourcesStats.children.length === 0 || 
                    sourcesStats.textContent.includes('Click \'Apply Filters\' to load data'))) {
                    console.log('[Progressive Loading] Sources tab - waiting for filter application');
                    loadingStates.priorityComplete = true;
                    hideRadarPulse();
                    return;
                }
                
                // Check if subtabs are loaded with actual content
                if (sourcesSubtabs) {
                    const hasContent = sourcesSubtabs.querySelector('.dash-tab-content') && 
                                     !sourcesSubtabs.textContent.includes('Apply filters to view data');
                    
                    if (hasContent) {
                        console.log('[Progressive Loading] Sources tab content loaded');
                        loadingStates.priorityComplete = true;
                        hideRadarPulse();
                        return;
                    }
                }
                
                // Still loading
                return;
            }
            
            // Special handling for Explore tab
            if (currentTab === 'explore') {
                // Check if sunburst is loaded
                const sunburst = document.getElementById('sunburst-chart');
                const sunburstLoaded = sunburst && 
                    sunburst.querySelector('.js-plotly-plot') && 
                    sunburst.querySelector('.js-plotly-plot').data && 
                    sunburst.querySelector('.js-plotly-plot').data.length > 0;
                
                // Check if chunks container has content (after clicking a segment)
                const chunksContainer = document.getElementById('explore-chunks-container');
                const chunksTitle = document.getElementById('chunks-selection-title');
                const hasChunks = chunksContainer && chunksContainer.children.length > 0;
                const hasTitle = chunksTitle && chunksTitle.textContent && chunksTitle.textContent.includes('Text Chunks for:');
                
                // If we have a title, we're loading chunks
                if (hasTitle) {
                    console.log('[Progressive Loading] Checking chunks - hasTitle:', hasTitle, 'hasChunks:', hasChunks);
                    
                    // More aggressive check - if we have chunks, consider it loaded
                    if (hasChunks) {
                        console.log('[Progressive Loading] Chunks loaded after segment click, hiding radar pulse');
                        loadingStates.priorityComplete = true;
                        hideRadarPulse();
                        
                        // Double-check and force hide any remaining loading indicators
                        setTimeout(() => {
                            const sweep = document.querySelector('.dash-loading-radar-sweep');
                            if (sweep) {
                                console.log('[Progressive Loading] Force removing radar sweep');
                                sweep.remove();
                            }
                        }, 100);
                        return;
                    }
                    // Still loading chunks
                    return;
                }
                
                // Initial load - just check sunburst
                if (sunburstLoaded) {
                    const isLoading = document.querySelector('[data-dash-is-loading="true"]') ||
                                    document.querySelector('._dash-loading:not(:empty)') ||
                                    document.querySelector('.dash-spinner:not(:empty)');
                    
                    if (!isLoading) {
                        console.log('[Progressive Loading] Initial sunburst loaded, hiding radar pulse');
                        loadingStates.priorityComplete = true;
                        hideRadarPulse();
                        return;
                    }
                }
                return; // Still loading
            }
            
            // For other tabs, check all priority components
            let allPriorityLoaded = true;
            
            for (const id of priorityIds) {
                const element = document.getElementById(id);
                if (element) {
                    const isLoading = element.getAttribute('data-dash-is-loading') === 'true' ||
                                    element.querySelector('._dash-loading') ||
                                    element.querySelector('.dash-spinner');
                    
                    if (isLoading) {
                        allPriorityLoaded = false;
                        break;
                    }
                }
            }
            
            if (allPriorityLoaded && priorityIds.length > 0) {
                console.log('[Progressive Loading] Priority components loaded, hiding radar pulse');
                loadingStates.priorityComplete = true;
                hideRadarPulse();
            }
        }
    }
    
    function hideRadarPulse() {
        const sweep = document.querySelector('.dash-loading-radar-sweep');
        if (sweep) {
            sweep.style.transition = 'opacity 0.5s ease-out';
            sweep.style.opacity = '0';
            setTimeout(() => sweep.remove(), 500);
        }
    }
    
    function showBackgroundLoadingIndicator() {
        // Create a subtle loading indicator in the corner
        const indicator = document.createElement('div');
        indicator.id = 'background-loading-indicator';
        indicator.innerHTML = `
            <div style="
                position: fixed;
                bottom: 20px;
                left: 20px;
                background: rgba(19, 55, 111, 0.9);
                color: white;
                padding: 10px 15px;
                border-radius: 4px;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 10px;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            ">
                <div class="spinner-border spinner-border-sm" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
                <span>Loading additional data...</span>
            </div>
        `;
        
        document.body.appendChild(indicator);
        
        // Check periodically if all loading is complete
        const checkInterval = setInterval(() => {
            if (window.dashLoadingMonitor) {
                const states = window.dashLoadingMonitor.getStates();
                if (states.components.length === 0 && 
                    states.graphs.length === 0 && 
                    states.callbacks.length === 0) {
                    
                    // All loading complete
                    const bgIndicator = document.getElementById('background-loading-indicator');
                    if (bgIndicator) {
                        bgIndicator.style.transition = 'opacity 0.5s ease-out';
                        bgIndicator.style.opacity = '0';
                        setTimeout(() => bgIndicator.remove(), 500);
                    }
                    clearInterval(checkInterval);
                }
            }
        }, 500);
        
        // Timeout after 30 seconds
        setTimeout(() => {
            clearInterval(checkInterval);
            const bgIndicator = document.getElementById('background-loading-indicator');
            if (bgIndicator) bgIndicator.remove();
        }, 30000);
    }
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupProgressiveLoading);
    } else {
        setupProgressiveLoading();
    }
    
    // Also check periodically for completion
    setInterval(function() {
        if (!loadingStates.priorityComplete) {
            checkPriorityCompletion();
        }
        
        // Extra check specifically for chunks
        const currentTab = getCurrentTab();
        if (currentTab === 'explore') {
            const chunksContainer = document.getElementById('explore-chunks-container');
            const chunksTitle = document.getElementById('chunks-selection-title');
            
            if (chunksContainer && chunksContainer.children.length > 0 && 
                chunksTitle && chunksTitle.textContent.includes('Text Chunks for:')) {
                
                // We have chunks! Hide the radar pulse
                const sweep = document.querySelector('.dash-loading-radar-sweep');
                if (sweep) {
                    console.log('[Progressive Loading] Chunks detected in periodic check, removing radar');
                    sweep.remove();
                    loadingStates.priorityComplete = true;
                }
            }
        }
        
        // Check Sources tab
        if (currentTab === 'sources') {
            const sourcesSubtabs = document.getElementById('sources-subtabs');
            if (sourcesSubtabs) {
                const hasRealContent = sourcesSubtabs.querySelector('.card') || 
                                     sourcesSubtabs.querySelector('canvas') ||
                                     sourcesSubtabs.querySelector('.js-plotly-plot');
                
                if (hasRealContent) {
                    const sweep = document.querySelector('.dash-loading-radar-sweep');
                    if (sweep) {
                        console.log('[Progressive Loading] Sources content detected, removing radar');
                        sweep.remove();
                        loadingStates.priorityComplete = true;
                    }
                }
            }
        }
    }, 250); // Check every 250ms
    
    // Expose for debugging
    window.progressiveLoading = {
        getPriorityComponents: () => PRIORITY_COMPONENTS,
        getLoadingStates: () => loadingStates,
        checkPriority: checkPriorityCompletion
    };
})();