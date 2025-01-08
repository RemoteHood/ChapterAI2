async function uploadChunks(file, chunkSize) {
    console.log("Starting chunk upload...");
    const totalChunks = Math.ceil(file.size / chunkSize);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const chunk = file.slice(start, end);

        const formData = new FormData();
        formData.append('chunk', chunk, `chunk_${i}.part`);
        formData.append('chunk_number', i);
        formData.append('total_chunks', totalChunks);

        const response = await fetch('/upload_chunk', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            console.error('Chunk upload failed');
            throw new Error('Chunk upload failed');
        }
        console.log(`Chunk ${i} uploaded successfully`);
    }
    alert('File uploaded successfully!');
}

document.getElementById('uploadButton').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const chunkSize = 1024 * 1024; // 1 MB chunks
        uploadChunks(file, chunkSize);
    } else {
        console.error('No file selected');
        alert('Please select a file to upload');
    }
});





