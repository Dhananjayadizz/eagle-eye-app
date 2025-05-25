// Create a placeholder image for the video feed
const placeholderSvg = document.createElement('div');
placeholderSvg.innerHTML = `
<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f8f9fa"/>
    <text x="50%" y="50%" font-family="Arial" font-size="20" text-anchor="middle" fill="#6c757d">Video feed will appear here</text>
    <text x="50%" y="55%" font-family="Arial" font-size="16" text-anchor="middle" fill="#6c757d">Select a camera and click Start Camera</text>
</svg>
`;

// Convert the SVG to a data URL
const svgString = new XMLSerializer().serializeToString(placeholderSvg.querySelector('svg'));
const encodedSvg = encodeURIComponent(svgString);
const dataUrl = `data:image/svg+xml;charset=utf-8,${encodedSvg}`;

// Set the placeholder image when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const processedImg = document.getElementById('processed-video-feed');
    if (processedImg) {
        processedImg.src = dataUrl;
    }
});
