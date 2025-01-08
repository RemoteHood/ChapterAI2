async function uploadChunks(file, chunkSize) {
    const totalChunks = Math.ceil(file.size / chunkSize);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const chunk = file.slice(start, end);

        const formData = new FormData();
        formData.append('chunk', chunk, `chunk_${i}.part`);
        formData.append('chunk_number', i);
        formData.append('total_chunks', totalChunks);

        const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Chunk upload failed');
        }
    }
    alert('File uploaded successfully!');
}

document.getElementById('uploadButton').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const chunkSize = 1024 * 1024; // 1 MB chunks
    uploadChunks(file, chunkSize);
});
