import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from werkzeug.utils import secure_filename
from logger import get_logger
from byaldi import RAGMultiModalModel

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

logger = get_logger(__name__)

# Configure upload folders
app.config['UPLOAD_FOLDER'] = 'uploaded_documents'
app.config['STATIC_FOLDER'] = 'static'
app.config['SESSION_FOLDER'] = 'sessions'
app.config['INDEX_FOLDER'] = os.path.join(os.getcwd(), '.byaldi')  # Set to .byaldi folder in current directory

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)

# Initialize global variables
RAG_models = {}  # Dictionary to store RAG models per session
app.config['INITIALIZATION_DONE'] = False  # Flag to track initialization
logger.info("Application started.")

def load_rag_model_for_session(session_id):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_path = os.path.join(app.config['INDEX_FOLDER'], session_id)

    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

def load_existing_indexes():
    """
    Loads all existing indexes from the .byaldi folder when the application starts.
    """
    global RAG_models
    if os.path.exists(app.config['INDEX_FOLDER']):
        for session_id in os.listdir(app.config['INDEX_FOLDER']):
            if os.path.isdir(os.path.join(app.config['INDEX_FOLDER'], session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")

@app.before_request
def initialize_app():
    """
    Initializes the application by loading existing indexes.
    This will run before the first request, but only once.
    """
    if not app.config['INITIALIZATION_DONE']:
        load_existing_indexes()
        app.config['INITIALIZATION_DONE'] = True
        logger.info("Application initialized and indexes loaded.")

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session_id = session['session_id']
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    # Load session data from file
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            chat_history = session_data.get('chat_history', [])
            session_name = session_data.get('session_name', 'Untitled Session')
            indexed_files = session_data.get('indexed_files', [])
    else:
        chat_history = []
        session_name = 'Untitled Session'
        indexed_files = []

    if request.method == 'POST':
        if 'upload' in request.form:
            # Handle file upload and indexing
            files = request.files.getlist('files')
            session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            os.makedirs(session_folder, exist_ok=True)
            uploaded_files = []
            for file in files:
                if file and file.filename:
                    filename = secure_filename(os.path.basename(file.filename))
                    file_path = os.path.join(session_folder, filename)
                    file.save(file_path)
                    uploaded_files.append(filename)
                    logger.info(f"File saved: {file_path}")
            try:
                index_name = session_id
                index_path = os.path.join(app.config['INDEX_FOLDER'], index_name)
                flash("Indexing documents. This may take a moment...", "info")
                RAG = index_documents(session_folder, index_name=index_name, index_path=index_path)
                RAG_models[session_id] = RAG
                session['index_name'] = index_name
                session['session_folder'] = session_folder
                indexed_files = os.listdir(session_folder)
                session_data = {
                    'session_name': session_name,
                    'chat_history': chat_history,
                    'indexed_files': indexed_files
                }
                with open(session_file, 'w') as f:
                    json.dump(session_data, f)
                flash("Documents indexed successfully.", "success")
                logger.info("Documents indexed successfully.")
            except Exception as e:
                logger.error(f"Error indexing documents: {e}")
                flash("An error occurred while indexing documents.", "danger")
            return redirect(url_for('chat'))
        elif 'send_query' in request.form:
            # Handle chat query
            try:
                query = request.form['query']
                if not query.strip():
                    flash("Please enter a query.", "warning")
                    return redirect(url_for('chat'))

                flash("Generating response. Please wait...", "info")

                model_choice = session.get('model', 'qwen')
                resized_height = int(session.get('resized_height', 280))
                resized_width = int(session.get('resized_width', 280))

                RAG = RAG_models.get(session_id)
                if not RAG:
                    load_rag_model_for_session(session_id)
                    RAG = RAG_models.get(session_id)
                    if not RAG:
                        flash("Please upload and index documents first.", "warning")
                        return redirect(url_for('chat'))

                images = retrieve_documents(RAG, query, session_id)
                response = generate_response(
                    images, query, session_id,
                    resized_height=resized_height,
                    resized_width=resized_width,
                    model_choice=model_choice
                )
                chat_entry = {
                    'user': query,
                    'response': response,
                    'images': images
                }
                chat_history.append(chat_entry)
                session_data = {
                    'session_name': session_name,
                    'chat_history': chat_history,
                    'indexed_files': indexed_files
                }
                with open(session_file, 'w') as f:
                    json.dump(session_data, f)
                logger.info("Response generated and added to chat history.")
                flash("Response generated.", "success")
                return redirect(url_for('chat'))
            except Exception as e:
                logger.error(f"Error in chat route: {e}")
                flash("An error occurred. Please try again.", "danger")
                return redirect(url_for('chat'))
        elif 'rename_session' in request.form:
            # Handle session renaming
            new_session_name = request.form.get('session_name', 'Untitled Session')
            session_name = new_session_name
            session_data = {
                'session_name': session_name,
                'chat_history': chat_history,
                'indexed_files': indexed_files
            }
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
            flash("Session name updated.", "success")
            return redirect(url_for('chat'))
        else:
            flash("Invalid request.", "warning")
            return redirect(url_for('chat'))
    else:
        session_files = os.listdir(app.config['SESSION_FOLDER'])
        chat_sessions = []
        for file in session_files:
            if file.endswith('.json'):
                s_id = file[:-5]
                with open(os.path.join(app.config['SESSION_FOLDER'], file), 'r') as f:
                    data = json.load(f)
                    name = data.get('session_name', 'Untitled Session')
                    chat_sessions.append({'id': s_id, 'name': name})

        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)

        return render_template('chat.html', chat_history=chat_history, chat_sessions=chat_sessions,
                               current_session=session_id, model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width,
                               session_name=session_name, indexed_files=indexed_files)

@app.route('/switch_session/<session_id>')
def switch_session(session_id):
    session['session_id'] = session_id
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    flash(f"Switched to session.", "info")
    return redirect(url_for('chat'))

@app.route('/rename_session', methods=['POST'])
def rename_session():
    session_id = session['session_id']
    new_session_name = request.form.get('new_session_name', 'Untitled Session')
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
    else:
        session_data = {}

    session_data['session_name'] = new_session_name

    with open(session_file, 'w') as f:
        json.dump(session_data, f)

    flash("Session name updated.", "success")
    return redirect(url_for('chat'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        model_choice = request.form.get('model', 'qwen')
        resized_height = request.form.get('resized_height', 280)
        resized_width = request.form.get('resized_width', 280)
        session['model'] = model_choice
        session['resized_height'] = resized_height
        session['resized_width'] = resized_width
        session.modified = True
        logger.info(f"Settings updated: model={model_choice}, resized_height={resized_height}, resized_width={resized_width}")
        flash("Settings updated.", "success")
        return redirect(url_for('chat'))
    else:
        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        return render_template('settings.html', model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width)

@app.route('/new_session')
def new_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
    session_data = {
        'session_name': session_name,
        'chat_history': [],
        'indexed_files': []
    }
    with open(session_file, 'w') as f:
        json.dump(session_data, f)
    flash("New chat session started.", "success")
    return redirect(url_for('chat'))

@app.route('/delete_session/<session_id>')
def delete_session(session_id):
    try:
        session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        global RAG_models
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        session_images_folder = os.path.join('static', 'images', session_id)
        if os.path.exists(session_images_folder):
            import shutil
            shutil.rmtree(session_images_folder)
        RAG_models.pop(session_id, None)
        if session.get('session_id') == session_id:
            session['session_id'] = str(uuid.uuid4())
        logger.info(f"Session {session_id} deleted.")
        flash("Session deleted.", "success")
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        flash("An error occurred while deleting the session.", "danger")
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(port=5050, debug=True)