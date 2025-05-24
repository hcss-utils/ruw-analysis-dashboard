// Custom loading script for smoother component loading
window.addEventListener('DOMContentLoaded', function() {
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
    
    // Spinner
    var spinner = document.createElement('div');
    spinner.style.width = '40px';
    spinner.style.height = '40px';
    spinner.style.margin = '0 auto';
    spinner.style.border = '4px solid #f3f3f3';
    spinner.style.borderTop = '4px solid #13376f'; // Match app theme
    spinner.style.borderRadius = '50%';
    spinner.style.animation = 'spin 1s linear infinite';
    
    // Add keyframes for spinner
    var style = document.createElement('style');
    style.textContent = '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
    document.head.appendChild(style);
    
    // Assemble loading indicator
    loadingDiv.appendChild(loadingText);
    loadingDiv.appendChild(spinner);
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