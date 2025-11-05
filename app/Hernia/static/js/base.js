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
});



// -----------------API DEL CHAT BOT---------------------

var Tawk_API=Tawk_API||{}, Tawk_LoadStart=new Date();
(function(){
var s1=document.createElement("script"),s0=document.getElementsByTagName("script")[0];
s1.async=true;
s1.src='https://embed.tawk.to/66ceb980ea492f34bc0ad75b/1i6bo97a1';
s1.charset='UTF-8';
s1.setAttribute('crossorigin','*');
s0.parentNode.insertBefore(s1,s0);
})();