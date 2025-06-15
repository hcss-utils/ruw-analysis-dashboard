// Responsive configuration for all Plotly graphs
(function() {
    'use strict';
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    function init() {
        // Configuration for responsive graphs
        const responsiveConfig = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'dashboard_export',
                height: 500,
                width: 700,
                scale: 1
            }
        };
        
        // Override Plotly newPlot to ensure responsive config
        if (window.Plotly) {
            const originalNewPlot = window.Plotly.newPlot;
            window.Plotly.newPlot = function(gd, data, layout, config) {
                // Merge responsive config with any provided config
                const finalConfig = Object.assign({}, responsiveConfig, config || {});
                
                // Ensure responsive layout
                const finalLayout = Object.assign({}, layout || {}, {
                    autosize: true,
                    margin: {
                        l: 50,
                        r: 50,
                        b: 50,
                        t: 50,
                        pad: 4
                    }
                });
                
                // Call original function with responsive settings
                return originalNewPlot.call(this, gd, data, finalLayout, finalConfig);
            };
            
            // Also override react method
            const originalReact = window.Plotly.react;
            window.Plotly.react = function(gd, data, layout, config) {
                const finalConfig = Object.assign({}, responsiveConfig, config || {});
                const finalLayout = Object.assign({}, layout || {}, {
                    autosize: true
                });
                return originalReact.call(this, gd, data, finalLayout, finalConfig);
            };
        }
        
        // Handle window resize
        let resizeTimeout;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(function() {
                // Resize all Plotly graphs
                const plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {
                    if (window.Plotly) {
                        window.Plotly.Plots.resize(plot);
                    }
                });
            }, 500);
        });
        
        // Handle orientation change on mobile
        window.addEventListener('orientationchange', function() {
            setTimeout(function() {
                const plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {
                    if (window.Plotly) {
                        window.Plotly.Plots.resize(plot);
                    }
                });
            }, 500);
        });
    }
    
    // Expose function to manually trigger resize
    window.resizeDashGraphs = function() {
        const plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach(function(plot) {
            if (window.Plotly) {
                window.Plotly.Plots.resize(plot);
            }
        });
    };
})();