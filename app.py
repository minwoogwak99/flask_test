from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_ollama import ChatOllama
import json
from llms import ollama_llm
# from langchain_community.document_loaders import WebBaseLoader

from various.get_text_from_link import get_from_link
from various.get_text_from_link_with_memory import get_from_link_with_memory

app = Flask(__name__)
CORS(app)


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

        response = ollama_llm.invoke(query)
        result = response.content

        return jsonify({"response": result}), 200
    except Exception as e:
        print(f"Error in /ai route: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/fromlink', methods=['POST'])
def fromLink():
    json_content = request.json
    if not json_content:
        return jsonify({"error": "No JSON data provided"}), 400

    query = json_content.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(type(query))
    prevLog = json_content.get('prevLog')

    test_link = ["https://lilianweng.github.io/posts/2023-06-23-agent/",
                 "https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign"]
    return Response(
        generate_stream(
            link=test_link, query=query['message'], prev_chat_log=prevLog),
        content_type='application/x-ndjson',  # Set content type to ndjson
        headers={'Transfer-Encoding': 'chunked'}
    )


def generate_stream(link, query, prev_chat_log):
    counter = 1
    for chunk in get_from_link_with_memory(link=link, input=query, prev_chat_log=prev_chat_log):
        if chunk:
            # Convert data to JSON and add newline
            data_json = json.dumps({counter: chunk})
            yield (data_json + "\n").encode('utf-8')
            counter += 1


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
