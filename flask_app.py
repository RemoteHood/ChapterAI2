from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_chunk():
    chunk = request.files['chunk']
    chunk_number = int(request.form['chunk_number'])
    total_chunks = int(request.form['total_chunks'])

    chunk_path = os.path.join(UPLOAD_DIR, f"chunk_{chunk_number}.part")
    with open(chunk_path, "wb") as chunk_file:
        chunk_file.write(chunk.read())

    # Check if all chunks are uploaded
    if len(os.listdir(UPLOAD_DIR)) == total_chunks:
        combine_chunks(UPLOAD_DIR, total_chunks)

    return jsonify({"status": "success"})

def combine_chunks(upload_dir, total_chunks):
    final_file_path = os.path.join(upload_dir, "final_file.pdf")
    with open(final_file_path, "wb") as final_file:
        for i in range(total_chunks):
            chunk_path = os.path.join(upload_dir, f"chunk_{i}.part")
            with open(chunk_path, "rb") as chunk_file:
                final_file.write(chunk_file.read())
            os.remove(chunk_path)

if __name__ == '__main__':
    app.run(debug=True)
