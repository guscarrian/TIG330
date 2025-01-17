from google import genai
from google.genai import types

from flask import Flask, request, jsonify, render_template, session
import pytesseract
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
import os
import json


# Configure Gemini API Key
client = genai.Client(api_key='you_api_key')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.urandom(24)  # Required for session handling



# Helper function to split text into chunks to fit the model's token limit
def chunk_text(text, max_tokens=3500):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:  # Add the last chunk if it exists
        chunks.append(" ".join(current_chunk))

    return chunks

# Helper function to convert summary into graph data
def generate_graph_data(summary, client):
    prompt = f"""
        You are an expert in extracting information and returning data as a JSON object.
        Given the following text, extract the key concepts and their relationships. 
        Return a JSON object and nothing else. The JSON object should contain the following keys: "nodes" and "edges".
        "nodes" should be a list of all the key concepts in the text. Each item in the list should be an object with two keys: 'id' and 'label'.
        "edges" should be a list of all the relationships between concepts. Each item in the list should be an object with two keys: 'source' and 'target'.
        Do not include any additional text other than the requested JSON object.
        Text: {summary}
        """
        
    response = client.models.generate_content(
            model='gemini-2.0-flash-exp', 
            contents=prompt
            )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "summary.txt"), 'w', encoding='utf-8') as summary_file:
        summary_file.write(summary)
    with open(os.path.join(log_dir, "llm_response.txt"), 'w', encoding='utf-8') as response_file:
        response_file.write(response.text)
    print("RAW LLM RESPONSE:", response.text)  # Print the raw response
    
    try:
        graph_data = json.loads(response.text)
        if "nodes" in graph_data and "edges" in graph_data:
            return graph_data
        else:
          print("Invalid JSON structure from LLM, returning empty graph")
          return {"nodes": [], "edges": []}
    except json.JSONDecodeError:
        print("Invalid JSON from LLM, returning empty graph")
        return {"nodes": [], "edges": []}


#Helper function to generate exam questions that will be used for the flashcards quiz
def generate_qa(summary, client):
    prompt = f"""
    You are an expert in creating exam questions and returning data as a JSON object.
    Given the following text, generate 5 possible exam questions with their corresponding answers.
    Return a JSON object and nothing else. The JSON object must have a "questions" key, which is an array of objects. 
    Each object in this array should have two keys: "question" and "answer". Do not include any additional text.
    Text: {summary}
    """
    
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt
    )

    print("RAW LLM RESPONSE FOR QA:", response.text)

    # Clean up the response text to ensure it's valid JSON
    cleaned_response = response.text.strip()  # Remove leading/trailing spaces
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].strip()

    try:
        qa_data = json.loads(cleaned_response)  # Loading the cleaned response as JSON
        # Validate JSON structure
        if 'questions' in qa_data and isinstance(qa_data['questions'], list):
            return qa_data
        
        # If validation fails, return an empty list
        print("Invalid JSON structure from LLM for QA generation")
        return {"questions": []}  # Return an empty list to prevent errors
        
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return {"questions": []}  # Return an empty list to prevent errors


# Helper function to generate quiz time data
def generate_quiz_data(summary, client):
    prompt = f"""
    You are an expert in creating interactive quizzes. Given the following summary, generate:
    1. Fill-in-the-blank questions with key terms as the answers. Provide three possible answers for each question: two incorrect options and one correct answer.
    For fill-in-the-blank questions, the format should be:
        {{
            "question": "The question text with a blank indicated as _________.",
            "answer_1": "An incorrect option.",
            "answer_2": "Another incorrect option.",
            "answer_correct": "The correct missing word."
        }}
    2. True/False questions based on the key points from the summary. Each question should be a statement with the option of "True" or "False" as the correct answer. The format for each question should be:
        {{
            "statement": "The statement to be evaluated as True or False.",
            "correct_answer": "True" or "False" (the correct answer)
        }}
    Return a JSON object with two keys: "fill_in_blanks" and "true_false_questions".
    Do not include any additional text or commentary.
    Summary: {summary}
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt
    )

    cleaned_response = response.text.strip()

    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].strip()

    try:
        quiz_data = json.loads(cleaned_response)
        if 'fill_in_blanks' in quiz_data and 'true_false_questions' in quiz_data:
            return quiz_data
        return {"fill_in_blanks": [], "true_false_questions": []}  # Return empty if structure is invalid
    except json.JSONDecodeError:
        return {"fill_in_blanks": [], "true_false_questions": []}  # Return empty in case of error



# Route to serve the frontend
@app.route('/')
def index():
    return render_template('frontend.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process file
    if filename.endswith('.pdf'):
        text = extract_text(filepath)
    else:
        text = pytesseract.image_to_string(filepath)

    # Split text into chunks
    chunks = chunk_text(text)

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        print(f"Processing chunk of size: {len(chunk)}")
        #print(f"Chunk being sent to model: {chunk}")
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp', 
            contents=f"Summarize the following text: {chunk}"
            )
        #summaries.append(response.result.strip())
        summaries.append(response.text.strip())
    
    
    # Combine all chunk summaries into one final summary
    final_summary = " ".join(summaries)

    # Generate graph data
    #graph_data = generate_graph_data(final_summary, client)
    #print('GRAPH DATA: ', graph_data)

    # Generate Q/A
    qa_data = generate_qa(final_summary, client)
    print('QA DATA: ', qa_data)

    # Generate quiz data
    quiz_data = generate_quiz_data(final_summary, client)
    print('QUIZ DATA: ', quiz_data)

    #NEW
    # Store data in the session for dynamic access
    session['summary'] = final_summary
    session['qa_data'] = qa_data
    session['quiz_data'] = quiz_data
    
    return jsonify({
        "summary": final_summary,
        #"interactive map": graph_data,
        "questions": qa_data["questions"],
        #"quiz": quiz_data["quiz"]
        "fill_in_blanks": quiz_data["fill_in_blanks"],
        "true_false_questions": quiz_data["true_false_questions"]

    })


# NEW
@app.route('/quiz', methods=['GET'])
def get_quiz():
    # Retrieve quiz data from the session
    quiz_data = session.get('quiz_data', None)

    if not quiz_data:
        return jsonify({"error": "No quiz data available. Please upload a file first."}), 404

    return jsonify(quiz_data)

# NEW
@app.route('/summary', methods=['GET'])
def get_summary():
    # Retrieve summary from the session
    summary = session.get('summary', None)

    if not summary:
        return jsonify({"error": "No summary available. Please upload a file first."}), 404

    return jsonify({"summary": summary})


if __name__ == '__main__':
    app.run(debug=True)