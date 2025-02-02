// Main JavaScript functionality for the web interface

document.addEventListener('DOMContentLoaded', function() {
    // File input handling
    const fileInput = document.getElementById('file');
    const fileLabel = document.querySelector('label[for="file"]');
    const form = document.querySelector('form');
    
    fileInput.addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name;
        if (fileName) {
            fileLabel.textContent = fileName;
        } else {
            fileLabel.textContent = 'Choose an X-ray image';
        }
    });
    
    // Form submission handling
    form.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        if (!file) {
            e.preventDefault();
            alert('Please select an image file');
            return;
        }
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            e.preventDefault();
            alert('Please select a valid image file (JPEG, JPG, or PNG)');
            return;
        }
        
        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.textContent = 'Analyzing...';
    });
});

// Handle image preview
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        
        reader.readAsDataURL(input.files[0]);
    }
}