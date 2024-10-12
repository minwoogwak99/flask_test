from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import ChatOllama
import bs4
# from langchain_community.document_loaders import WebBaseLoader


app = Flask(__name__)
CORS(app)

cached_llm = ChatOllama(model="llama3.2")


@app.route('/post', methods=['GET'])
def hello_world():
    # Check if the request is a POST request
    if request.method == 'GET':
        # Return a JSON response
        return jsonify(message="hello world"), 200


@app.route("/ai", methods=["POST"])
def aiPost():
    print('post /ai called')
    try:
        json_content = request.json
        if not json_content:
            return jsonify({"error": "No JSON data provided"}), 400

        query = json_content.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400

        response = cached_llm.invoke(query)
        result = response.content

        return jsonify({"response": result}), 200
    except Exception as e:
        print(f"Error in /ai route: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/pdf', methods=['POST'])
def pdfPost():
    file = request.files['file']
    file_name = file.filename
    save_dir = "pdf/" + file_name
    file.save(save_dir)
    print(f"filename: {file_name}")

    response = {"status": "Successfully Uploaded", "filename": file_name}
    return jsonify(response), 200


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
