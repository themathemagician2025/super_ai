/**
 * Lazy Loader for JavaScript and Images
 *
 * This module provides functionality to lazy load JavaScript files
 * and images to improve initial page load performance.
 */

/**
 * Lazy load a JavaScript file
 *
 * @param {string} src - The source URL of the script
 * @param {Function} callback - Optional callback function to execute after loading
 * @param {Object} options - Optional configuration options
 * @returns {Promise} - Promise that resolves when the script is loaded
 */
export function lazyLoadScript(src, callback = null, options = {}) {
    return new Promise((resolve, reject) => {
        // Check if script already exists
        const existingScript = document.querySelector(`script[src="${src}"]`);
        if (existingScript) {
            if (callback) callback();
            resolve(existingScript);
            return;
        }

        // Create script element
        const script = document.createElement('script');
        script.src = src;
        script.async = options.async !== false;

        // Add any additional attributes
        if (options.attrs) {
            Object.entries(options.attrs).forEach(([key, value]) => {
                script.setAttribute(key, value);
            });
        }

        // Handle loading events
        script.onload = () => {
            if (callback) callback();
            resolve(script);
        };

        script.onerror = (error) => {
            reject(new Error(`Error loading script: ${src}`));
            if (options.removeOnError) {
                document.head.removeChild(script);
            }
        };

        // Add to document
        document.head.appendChild(script);
    });
}

/**
 * Lazy load multiple JavaScript files
 *
 * @param {Array<string>} sources - Array of script source URLs
 * @param {Function} callback - Optional callback function to execute after all scripts are loaded
 * @param {Object} options - Optional configuration options
 * @returns {Promise} - Promise that resolves when all scripts are loaded
 */
export function lazyLoadScripts(sources, callback = null, options = {}) {
    const promises = sources.map(src => lazyLoadScript(src, null, options));

    return Promise.all(promises)
        .then(scripts => {
            if (callback) callback(scripts);
            return scripts;
        });
}

/**
 * Observe elements with data-src attribute and lazy load them
 * when they enter the viewport
 */
export function initLazyImages() {
    // Check if IntersectionObserver is supported
    if (!('IntersectionObserver' in window)) {
        // Fallback for older browsers
        document.querySelectorAll('img[data-src]').forEach(img => {
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
        });
        return;
    }

    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;

                // Replace src with data-src
                img.src = img.dataset.src;

                // Handle loading and error events
                img.onload = () => {
                    img.removeAttribute('data-src');
                    img.classList.add('loaded');
                    observer.unobserve(img);
                };

                img.onerror = () => {
                    img.src = img.dataset.fallback || '';
                    observer.unobserve(img);
                };
            }
        });
    }, {
        rootMargin: '200px 0px',  // Start loading when image is 200px from viewport
        threshold: 0.01
    });

    // Observe all images with data-src attribute
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

/**
 * Initialize lazy loading when the DOM is ready
 */
export function initLazyLoading() {
    // Initialize lazy image loading
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initLazyImages);
    } else {
        initLazyImages();
    }

    // Dynamically load non-critical scripts
    window.addEventListener('load', () => {
        // Example: Load analytics after page load
        setTimeout(() => {
            lazyLoadScript('/static/js/analytics.js');
        }, 2000);
    });
}

// Auto-initialize if not imported as a module
if (typeof window !== 'undefined') {
    initLazyLoading();
}

export default {
    lazyLoadScript,
    lazyLoadScripts,
    initLazyImages,
    initLazyLoading
};
