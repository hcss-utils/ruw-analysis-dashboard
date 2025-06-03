// Comprehensive Dash loading state monitor
(function() {
    'use strict';
    
    // Configuration
    const DEBUG = true; // Set to false in production
    const SWEEP_CLEANUP_DELAY = 500; // ms to wait after last loading element disappears
    
    // State tracking
    const loadingStates = {
        components: new Map(), // Track individual component loading states
        graphs: new Map(),     // Track graph-specific loading
        callbacks: new Set(),  // Track active callbacks
        lastActivity: Date.now()
    };
    
    // Logging helper
    function log(...args) {
        if (DEBUG) {
            console.log('[Loading Monitor]', ...args);
        }
    }
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    function init() {
        log('Initializing loading monitor');
        
        // Override Dash's loading state management
        interceptDashLoading();
        
        // Set up mutation observer for DOM changes
        setupMutationObserver();
        
        // Monitor plotly graph loading
        monitorPlotlyGraphs();
        
        // Add CSS for radar sweep if not present
        addRadarSweepStyles();
    }
    
    // Intercept Dash's loading state changes
    function interceptDashLoading() {
        // Hook into Dash's internal loading state
        if (window.dash_clientside && window.dash_clientside.set_props) {
            const originalSetProps = window.dash_clientside.set_props;
            window.dash_clientside.set_props = function(id, props) {
                // Check for loading-related prop changes
                if (props && typeof props === 'object') {
                    if ('data-dash-is-loading' in props || 'is_loading' in props) {
                        const isLoading = props['data-dash-is-loading'] || props['is_loading'];
                        log(`Component ${id} loading state:`, isLoading);
                        
                        if (isLoading) {
                            loadingStates.components.set(id, true);
                            showRadarSweep();
                        } else {
                            loadingStates.components.delete(id);
                            checkAndHideRadarSweep();
                        }
                    }
                }
                
                return originalSetProps.apply(this, arguments);
            };
        }
        
        // Hook into callback execution
        if (window.dash_clientside && window.dash_clientside.callback_context) {
            const originalContext = window.dash_clientside.callback_context;
            Object.defineProperty(window.dash_clientside, 'callback_context', {
                get: function() {
                    const context = originalContext;
                    if (context && context.triggered && context.triggered.length > 0) {
                        log('Callback triggered:', context.triggered);
                        loadingStates.callbacks.add(context.triggered[0].prop_id);
                        showRadarSweep();
                        
                        // Remove from tracking after callback completes
                        setTimeout(() => {
                            loadingStates.callbacks.delete(context.triggered[0].prop_id);
                            checkAndHideRadarSweep();
                        }, 100);
                    }
                    return context;
                },
                set: function(value) {
                    originalContext = value;
                }
            });
        }
    }
    
    // Set up mutation observer
    function setupMutationObserver() {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                // Check for loading-related attribute changes
                if (mutation.type === 'attributes') {
                    const target = mutation.target;
                    
                    // Check data-dash-is-loading attribute
                    if (mutation.attributeName === 'data-dash-is-loading') {
                        const isLoading = target.getAttribute('data-dash-is-loading') === 'true';
                        const id = target.id || target.className || 'unknown';
                        
                        log(`Element ${id} loading attribute:`, isLoading);
                        
                        if (isLoading) {
                            loadingStates.components.set(id, true);
                            showRadarSweep();
                        } else {
                            loadingStates.components.delete(id);
                            checkAndHideRadarSweep();
                        }
                    }
                    
                    // Check for Dash loading classes
                    if (mutation.attributeName === 'class') {
                        const classList = target.classList;
                        const id = target.id || 'class-based-' + Date.now();
                        
                        if (classList.contains('_dash-loading') || 
                            classList.contains('_dash-loading-callback') ||
                            classList.contains('dash-spinner')) {
                            log(`Element ${id} has loading class`);
                            loadingStates.components.set(id, true);
                            showRadarSweep();
                        } else if (loadingStates.components.has(id)) {
                            log(`Element ${id} removed loading class`);
                            loadingStates.components.delete(id);
                            checkAndHideRadarSweep();
                        }
                    }
                }
                
                // Check for added/removed nodes
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === 1) {
                            checkNodeForLoading(node);
                        }
                    });
                    
                    mutation.removedNodes.forEach(node => {
                        if (node.nodeType === 1) {
                            removeNodeFromTracking(node);
                        }
                    });
                }
            });
        });
        
        // Start observing
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['data-dash-is-loading', 'class', 'style']
        });
    }
    
    // Check a node and its children for loading states
    function checkNodeForLoading(node) {
        if (!node.querySelectorAll) return;
        
        // Check the node itself
        if (node.getAttribute('data-dash-is-loading') === 'true' ||
            node.classList.contains('_dash-loading') ||
            node.classList.contains('_dash-loading-callback') ||
            node.classList.contains('dash-spinner')) {
            
            const id = node.id || node.className || 'node-' + Date.now();
            log(`Found loading node: ${id}`);
            loadingStates.components.set(id, true);
            showRadarSweep();
        }
        
        // Check for plotly graphs
        if (node.classList && node.classList.contains('js-plotly-plot')) {
            const id = node.id || 'graph-' + Date.now();
            log(`Found Plotly graph: ${id}`);
            trackPlotlyGraph(node);
        }
        
        // Check children
        const loadingElements = node.querySelectorAll('[data-dash-is-loading="true"], ._dash-loading, ._dash-loading-callback, .dash-spinner, .js-plotly-plot');
        loadingElements.forEach(el => {
            const id = el.id || el.className || 'child-' + Date.now();
            if (el.classList && el.classList.contains('js-plotly-plot')) {
                trackPlotlyGraph(el);
            } else {
                loadingStates.components.set(id, true);
                showRadarSweep();
            }
        });
    }
    
    // Remove node from tracking
    function removeNodeFromTracking(node) {
        const id = node.id || node.className;
        if (id && loadingStates.components.has(id)) {
            log(`Removing node from tracking: ${id}`);
            loadingStates.components.delete(id);
            checkAndHideRadarSweep();
        }
        
        if (id && loadingStates.graphs.has(id)) {
            log(`Removing graph from tracking: ${id}`);
            loadingStates.graphs.delete(id);
            checkAndHideRadarSweep();
        }
    }
    
    // Monitor Plotly graph loading
    function monitorPlotlyGraphs() {
        // Override Plotly's newPlot to track graph rendering
        if (window.Plotly && window.Plotly.newPlot) {
            const originalNewPlot = window.Plotly.newPlot;
            window.Plotly.newPlot = function(gd, data, layout, config) {
                const id = (typeof gd === 'string') ? gd : (gd.id || 'plotly-' + Date.now());
                log(`Plotly.newPlot called for ${id}`);
                
                loadingStates.graphs.set(id, true);
                showRadarSweep();
                
                // Call original function
                const result = originalNewPlot.apply(this, arguments);
                
                // Mark as loaded after rendering completes
                if (result && result.then) {
                    result.then(() => {
                        log(`Plotly graph ${id} rendered`);
                        loadingStates.graphs.delete(id);
                        checkAndHideRadarSweep();
                    });
                } else {
                    // Fallback for synchronous rendering
                    setTimeout(() => {
                        log(`Plotly graph ${id} rendered (timeout)`);
                        loadingStates.graphs.delete(id);
                        checkAndHideRadarSweep();
                    }, 100);
                }
                
                return result;
            };
        }
        
        // Also monitor react and redraw
        if (window.Plotly && window.Plotly.react) {
            const originalReact = window.Plotly.react;
            window.Plotly.react = function(gd, data, layout, config) {
                const id = (typeof gd === 'string') ? gd : (gd.id || 'plotly-react-' + Date.now());
                log(`Plotly.react called for ${id}`);
                
                loadingStates.graphs.set(id, true);
                showRadarSweep();
                
                const result = originalReact.apply(this, arguments);
                
                if (result && result.then) {
                    result.then(() => {
                        log(`Plotly graph ${id} updated`);
                        loadingStates.graphs.delete(id);
                        checkAndHideRadarSweep();
                    });
                } else {
                    setTimeout(() => {
                        log(`Plotly graph ${id} updated (timeout)`);
                        loadingStates.graphs.delete(id);
                        checkAndHideRadarSweep();
                    }, 100);
                }
                
                return result;
            };
        }
    }
    
    // Track individual Plotly graph
    function trackPlotlyGraph(graphElement) {
        const id = graphElement.id || 'graph-element-' + Date.now();
        
        // Check if graph is already rendered
        if (graphElement.data && graphElement.layout) {
            log(`Graph ${id} already rendered`);
            return;
        }
        
        log(`Tracking graph ${id}`);
        loadingStates.graphs.set(id, true);
        showRadarSweep();
        
        // Monitor for graph completion
        const checkInterval = setInterval(() => {
            if (graphElement.data && graphElement.layout) {
                log(`Graph ${id} completed`);
                clearInterval(checkInterval);
                loadingStates.graphs.delete(id);
                checkAndHideRadarSweep();
            }
        }, 50);
        
        // Timeout fallback
        setTimeout(() => {
            clearInterval(checkInterval);
            if (loadingStates.graphs.has(id)) {
                log(`Graph ${id} timeout`);
                loadingStates.graphs.delete(id);
                checkAndHideRadarSweep();
            }
        }, 5000);
    }
    
    // Show radar sweep
    function showRadarSweep() {
        loadingStates.lastActivity = Date.now();
        
        if (!document.querySelector('.dash-loading-radar-sweep')) {
            const container = document.createElement('div');
            container.className = 'dash-loading-radar-sweep';
            
            const overlay = document.createElement('div');
            overlay.className = 'dash-loading-overlay';
            
            const sweep = document.createElement('div');
            sweep.className = 'dash-loading-sweep-band';
            
            container.appendChild(overlay);
            container.appendChild(sweep);
            document.body.appendChild(container);
            
            log('Radar sweep shown');
        }
    }
    
    // Check if we should hide radar sweep
    function checkAndHideRadarSweep() {
        const totalLoading = loadingStates.components.size + 
                           loadingStates.graphs.size + 
                           loadingStates.callbacks.size;
        
        log(`Active loading states: ${loadingStates.components.size} components, ${loadingStates.graphs.size} graphs, ${loadingStates.callbacks.size} callbacks`);
        
        if (totalLoading === 0) {
            // Wait a bit to ensure no new loading states appear
            setTimeout(() => {
                const stillLoading = loadingStates.components.size + 
                                   loadingStates.graphs.size + 
                                   loadingStates.callbacks.size;
                
                if (stillLoading === 0) {
                    hideRadarSweep();
                }
            }, SWEEP_CLEANUP_DELAY);
        }
    }
    
    // Hide radar sweep
    function hideRadarSweep() {
        const sweep = document.querySelector('.dash-loading-radar-sweep');
        if (sweep) {
            sweep.remove();
            log('Radar sweep hidden');
        }
    }
    
    // Add radar sweep styles
    function addRadarSweepStyles() {
        if (!document.querySelector('#dash-loading-radar-styles')) {
            const style = document.createElement('style');
            style.id = 'dash-loading-radar-styles';
            style.textContent = `
                .dash-loading-radar-sweep {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    pointer-events: none;
                    z-index: 9999;
                    overflow: hidden;
                }
                
                .dash-loading-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(255, 255, 255, 0.3);
                }
                
                .dash-loading-sweep-band {
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
                    animation: dash-radar-sweep 2s ease-in-out infinite;
                }
                
                @keyframes dash-radar-sweep {
                    0% { left: -100%; }
                    100% { left: 100%; }
                }
                
                /* Hide default Dash loading indicators */
                ._dash-loading,
                ._dash-loading-callback,
                .dash-spinner {
                    display: none !important;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Expose for debugging
    window.dashLoadingMonitor = {
        getStates: () => ({
            components: Array.from(loadingStates.components.keys()),
            graphs: Array.from(loadingStates.graphs.keys()),
            callbacks: Array.from(loadingStates.callbacks)
        }),
        forceHide: hideRadarSweep,
        forceShow: showRadarSweep
    };
})();