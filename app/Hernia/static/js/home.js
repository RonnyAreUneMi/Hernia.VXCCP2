document.addEventListener('DOMContentLoaded', function() {
    const alertBox = document.getElementById('customAlert');
    const closeButton = alertBox.querySelector('.close-btn');

    if (alertBox) {
        // Mostrar el alert con animación de entrada
        alertBox.classList.remove('translate-x-full', 'opacity-0');
        alertBox.classList.add('translate-x-0', 'opacity-100');

        // Ocultar el alert después de 5 segundos
        setTimeout(() => {
            alertBox.classList.add('translate-x-full', 'opacity-0');
            setTimeout(() => alertBox.classList.add('hidden'), 300); // Esperar a que termine la animación
        }, 5000);

        // Cerrar el alert al hacer clic en el botón
        closeButton.addEventListener('click', () => {
            alertBox.classList.add('translate-x-full', 'opacity-0');
            setTimeout(() => alertBox.classList.add('hidden'), 300);
        });
    }
});