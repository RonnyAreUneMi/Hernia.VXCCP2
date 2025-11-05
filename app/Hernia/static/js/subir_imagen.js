document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const fileInput = form.querySelector('input[type="file"]');

    form.addEventListener('submit', (event) => {
        const file = fileInput.files[0];
        if (file) {
            const fileType = file.type;
            const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];

            if (!allowedTypes.includes(fileType)) {
                event.preventDefault(); // Previene el envío del formulario
                Swal.fire({
                    icon: 'error',
                    title: 'Error',
                    text: 'Por favor, selecciona una imagen en formato PNG, JPG o JPEG.',
                    confirmButtonText: 'OK'
                });
            }
        }
    });
});