
function toggleDarkMode() {
    const body = document.getElementById('body');
    const isDarkMode = body.classList.toggle('dark-mode');
    
    if (isDarkMode) {
        body.classList.remove('light-mode');
    } else {
        body.classList.add('light-mode');
    }

    localStorage.setItem('darkMode', isDarkMode);
}

document.addEventListener('DOMContentLoaded', () => {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkMode = localStorage.getItem('darkMode') === 'true';
    const body = document.getElementById('body');
    
    if (darkMode) {
        body.classList.add('dark-mode');
        darkModeToggle.checked = true;
    } else {
        body.classList.add('light-mode');
        darkModeToggle.checked = false;
    }

    darkModeToggle.addEventListener('change', toggleDarkMode);

    const form = document.querySelector('form');
    const overlay = document.getElementById('loadingScreen');

    form.addEventListener('submit', function() {
        overlay.classList.add('show');
    });
});
