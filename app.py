import streamlit as st
import json
import os

import faiss
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT / GEMINI
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# GEMINI MODELS
# ============================================================

models_to_try = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Lab Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Lab Assistant")

st.write(
    "Intelligent Electronics and Communication Systems "
    "Experiment Diagnostic Assistant"
)


# ============================================================
# LOAD EXISTING KNOWLEDGE BASE
# ============================================================

try:

    with open("knowledge_base.json", "r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

except Exception as e:

    st.error(f"Could not load knowledge_base.json: {e}")
    st.stop()


# ============================================================
# LOAD EXISTING FAULT DATASET
# ============================================================

try:

    with open("fault_dataset.json", "r", encoding="utf-8") as file:
        fault_dataset = json.load(file)

except Exception as e:

    st.error(f"Could not load fault_dataset.json: {e}")
    st.stop()


# ============================================================
# LOAD COMMUNICATION SYSTEMS RAG DATABASE
# ============================================================

RAG_PATH = "experiments/combined_database"

try:

    # Load FAISS index
    faiss_index = faiss.read_index(
        os.path.join(RAG_PATH, "faiss.index")
    )

    # Load chunks
    with open(
        os.path.join(RAG_PATH, "chunks.json"),
        "r",
        encoding="utf-8"
    ) as file:

        rag_chunks = json.load(file)

    # Load procedure database
    with open(
        os.path.join(RAG_PATH, "procedure_database.json"),
        "r",
        encoding="utf-8"
    ) as file:

        procedure_database = json.load(file)

    # Load embedding model
    embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True
)

except Exception as e:

    st.error(
        "Communication Systems RAG database could not be loaded."
    )

    st.error(str(e))

    st.stop()


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

communication_experiments = {

    "1": {
        "name": "Amplitude Modulation and Demodulation",
        "short_name": "AM"
    },

    "2": {
        "name": "Frequency Modulation",
        "short_name": "FM"
    },

    "3": {
        "name": "Pulse Amplitude Modulation and Demodulation",
        "short_name": "PAM"
    },

    "4": {
        "name": "Pulse Width Modulation",
        "short_name": "PWM"
    },

    "5": {
        "name": "Flat Top Sampling",
        "short_name": "Flat Top Sampling"
    },

    "6": {
        "name": "Amplitude Shift Keying",
        "short_name": "ASK"
    },

    "7": {
        "name": "Phase Shift Keying",
        "short_name": "PSK"
    }
}


# ============================================================
# RAG SEARCH FUNCTION
# ============================================================

def search_lab_manual(query, k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(
        query_embedding,
        k
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        chunk = rag_chunks[int(index)]

        results.append({

            "score": float(score),

            "experiment_number":
                chunk["experiment_number"],

            "experiment":
                chunk["experiment_name"],

            "page":
                chunk["manual_page"],

            "text":
                chunk["text"]

        })

    return results


# ============================================================
# DETECT COMMUNICATION SYSTEMS EXPERIMENT
# ============================================================

def detect_communication_experiment(question):

    question_lower = question.lower()

    keywords = {

        "1": [
            "amplitude modulation",
            "am modulation",
            "am"
        ],

        "2": [
            "frequency modulation",
            "fm modulation"
        ],

        "3": [
            "pulse amplitude modulation",
            "pam modulation",
            "pam"
        ],

        "4": [
            "pulse width modulation",
            "pwm modulation",
            "pwm"
        ],

        "5": [
            "flat top sampling",
            "flat top"
        ],

        "6": [
            "amplitude shift keying",
            "ask modulation",
            "ask"
        ],

        "7": [
            "phase shift keying",
            "psk modulation",
            "psk"
        ]

    }

    # Longer keywords first
    all_keywords = []

    for experiment_number, words in keywords.items():

        for word in words:

            all_keywords.append(
                (word, experiment_number)
            )

    all_keywords.sort(
        key=lambda x: len(x[0]),
        reverse=True
    )

    for word, experiment_number in all_keywords:

        if word in question_lower:

            return experiment_number

    return None


# ============================================================
# PROCEDURE SEARCH
# ============================================================

def get_procedure(experiment_number):

    experiment_number = str(experiment_number)

    if experiment_number in procedure_database:

        return procedure_database[experiment_number]

    return None


# ============================================================
# GEMINI RAG ANSWER
# ============================================================

def answer_from_lab_manual(question):

    experiment_number = detect_communication_experiment(
        question
    )

    # --------------------------------------------------------
    # PROCEDURE QUESTION
    # --------------------------------------------------------

    is_procedure = any(
        word in question.lower()
        for word in [
            "procedure",
            "steps",
            "how to perform",
            "how to do"
        ]
    )

    if (
        is_procedure
        and experiment_number is not None
    ):

        procedure_data = get_procedure(
            experiment_number
        )

        if procedure_data is not None:

            context = json.dumps(
                procedure_data,
                indent=2
            )

        else:

            results = search_lab_manual(
                question,
                k=5
            )

            context = "\n\n".join(
                [
                    result["text"]
                    for result in results
                ]
            )

    else:

        results = search_lab_manual(
            question,
            k=5
        )

        context = "\n\n".join(
            [
                f"""
Experiment: {result['experiment']}
Manual Page: {result['page']}

{result['text']}
"""
                for result in results
            ]
        )

    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant for a Communication Systems laboratory.

The laboratory manual is the SOURCE OF TRUTH.

Student question:
{question}

Relevant information retrieved from the laboratory manual:

{context}

IMPORTANT RULES:

1. Answer using the retrieved laboratory manual information.
2. Do not invent information that is not supported by the manual.
3. Do not change numerical values from the manual.
4. Do not change formulas from the manual.
5. Preserve procedure order when explaining procedures.
6. If OCR text is unclear, say that the manual text is unclear.
7. Do not guess missing values.
8. Keep the explanation simple and suitable for a laboratory student.
9. Mention the experiment and manual page when useful.

Answer the student's question clearly.
"""

    response = None

    for model_name in models_to_try:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            break

        except Exception:
            continue

    if response:

        return response.text

    return None


# ============================================================
# MODE SELECTION
# ============================================================

st.sidebar.header("🔧 Assistant Mode")

mode = st.sidebar.radio(
    "Choose what you want to do:",
    [
        "Existing Electronics Experiments",
        "Communication Systems Lab"
    ]
)


# ============================================================
# EXISTING ELECTRONICS EXPERIMENTS
# ============================================================

if mode == "Existing Electronics Experiments":

    st.header("🔌 Electronics Experiment Diagnosis")

    experiment = st.selectbox(
        "Select your experiment:",
        [
            "Band Pass Filter",
            "Voltage Divider",
            "LED Circuit"
        ]
    )

    experiment_key = experiment.lower().replace(
        " ",
        "_"
    )

    data = knowledge_base[experiment_key]


    # --------------------------------------------------------
    # EXPERIMENT INFORMATION
    # --------------------------------------------------------

    st.subheader("Experiment Information")

    st.write(
        "**Formula:**",
        data["formula"]
    )

    st.write(
        "**Expected Result:**",
        data["expected_result"]
    )


    # --------------------------------------------------------
    # COMMON FAULTS
    # --------------------------------------------------------

    st.subheader("Common Faults")

    for fault in data["common_faults"]:

        st.write(
            "•",
            fault
        )


    # --------------------------------------------------------
    # CIRCUIT PHOTO
    # --------------------------------------------------------

    st.subheader("Circuit Photo")

    uploaded_image = st.file_uploader(
        "Upload a photo of your circuit:",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="existing_circuit_image"
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Circuit",
            width="stretch"
        )


    # --------------------------------------------------------
    # MEASUREMENTS
    # --------------------------------------------------------

    st.subheader("📊 Measurements")

    input_voltage = st.number_input(
        "Input Voltage (V):",
        min_value=0.0,
        step=0.1
    )

    output_voltage = st.number_input(
        "Output Voltage (V):",
        min_value=0.0,
        step=0.1
    )

    frequency = st.number_input(
        "Frequency (Hz):",
        min_value=0.0,
        step=1.0
    )


    # --------------------------------------------------------
    # OBSERVED RESULT
    # --------------------------------------------------------

    st.subheader("Observed Result")

    observed_result = st.text_input(
        "Enter your measured result:",
        placeholder="Example: 15 kHz"
    )


    # --------------------------------------------------------
    # CIRCUIT DIAGNOSIS
    # --------------------------------------------------------

    st.divider()

    st.header("🔍 Circuit Diagnosis")

    if st.button(
        "Diagnose Circuit",
        key="diagnose_existing"
    ):

        if not uploaded_image:

            st.warning(
                "Please upload a circuit photo."
            )

        elif not observed_result:

            st.warning(
                "Please enter your observed result."
            )

        else:

            with st.spinner(
                "AI is analyzing your circuit..."
            ):

                training_cases = fault_dataset.get(
                    experiment_key,
                    []
                )

                training_information = json.dumps(
                    training_cases,
                    indent=2
                )

                image_part = types.Part.from_bytes(
                    data=uploaded_image.getvalue(),
                    mime_type=uploaded_image.type
                )

                prompt = f"""
You are an electronics laboratory diagnostic assistant.

Your task is to diagnose faults in a student's
electronics experiment.

EXPERIMENT:
{experiment}

EXPERIMENT FORMULA:
{data["formula"]}

EXPECTED RESULT:
{data["expected_result"]}

COMMON FAULTS:
{", ".join(data["common_faults"])}

STUDENT MEASUREMENTS:

Input Voltage:
{input_voltage} V

Output Voltage:
{output_voltage} V

Frequency:
{frequency} Hz

Student's observed result:
{observed_result}

REFERENCE FAULT CASES:

{training_information}

IMPORTANT INSTRUCTIONS:

1. Analyze the student's actual circuit image.
2. Analyze the student's measurements.
3. Use the reference fault cases as diagnostic knowledge.
4. Do not automatically select a reference fault just because it exists.
5. Compare the student's symptoms with possible faults.
6. If the photograph is unclear, clearly say that the wiring cannot be confirmed.
7. Never claim that a connection is definitely wrong if the image does not clearly show it.
8. Give the most likely fault first.
9. Give alternative possible faults when appropriate.
10. Explain what the student should physically check.

Give your answer using exactly these sections:

1. MOST LIKELY FAULT

2. WHY THIS MAY BE HAPPENING

3. EVIDENCE FROM THE CIRCUIT

4. WHAT TO CHECK

5. CORRECTIVE ACTION

6. EXPECTED RESULT

7. CONFIDENCE

Keep the explanation simple for an electronics laboratory student.
"""

                response = None

                for model_name in models_to_try:

                    try:

                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                prompt,
                                image_part
                            ]
                        )

                        break

                    except Exception:
                        continue


                if response:

                    st.subheader(
                        "🤖 AI Diagnosis"
                    )

                    st.write(
                        response.text
                    )

                else:

                    st.error(
                        "Gemini is currently unavailable. "
                        "Please try again later."
                    )


    # --------------------------------------------------------
    # EXISTING ASK AI
    # --------------------------------------------------------

    st.divider()

    st.header(
        "💬 Ask AI About This Experiment"
    )

    question = st.text_input(
        "Your question:",
        placeholder="Example: Why is my output voltage different?",
        key="existing_question"
    )

    if st.button(
        "Ask AI",
        key="existing_ask_ai"
    ):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is thinking..."
            ):

                training_cases = fault_dataset.get(
                    experiment_key,
                    []
                )

                training_information = json.dumps(
                    training_cases,
                    indent=2
                )

                chat_prompt = f"""
You are an electronics laboratory teaching assistant.

The student is studying:

Experiment:
{experiment}

Formula:
{data["formula"]}

Expected result:
{data["expected_result"]}

Common faults:
{", ".join(data["common_faults"])}

Reference fault cases:
{training_information}

Student's question:
{question}

Use the experiment information and reference fault cases
to answer the student's question.

Do not invent measurements or circuit connections.

Answer clearly and simply.
"""

                response = None

                for model_name in models_to_try:

                    try:

                        response = client.models.generate_content(
                            model=model_name,
                            contents=chat_prompt
                        )

                        break

                    except Exception:
                        continue


                if response:

                    st.subheader(
                        "🤖 AI Answer"
                    )

                    st.write(
                        response.text
                    )

                else:

                    st.error(
                        "Gemini is currently unavailable. "
                        "Please try again later."
                    )


# ============================================================
# COMMUNICATION SYSTEMS LAB
# ============================================================

else:

    st.header(
        "📡 Communication Systems Laboratory"
    )

    st.write(
        "Ask questions about the 7 Communication Systems "
        "experiments using the laboratory manual."
    )


    # --------------------------------------------------------
    # SELECT EXPERIMENT
    # --------------------------------------------------------

    selected_experiment_number = st.selectbox(

        "Select Communication Systems experiment:",

        options=list(
            communication_experiments.keys()
        ),

        format_func=lambda x:
            f"Experiment {x} — "
            f"{communication_experiments[x]['name']}"
    )


    selected_experiment = communication_experiments[
        selected_experiment_number
    ]


    st.info(
        f"Selected: Experiment "
        f"{selected_experiment_number} — "
        f"{selected_experiment['name']}"
    )


    # --------------------------------------------------------
    # PROCEDURE BUTTON
    # --------------------------------------------------------

    st.subheader(
        "📋 Experiment Procedure"
    )

    if st.button(
        "Show Procedure",
        key="show_procedure"
    ):

        procedure = get_procedure(
            selected_experiment_number
        )

        if procedure:

            st.write(
                procedure
            )

        else:

            st.warning(
                "Procedure information not found."
            )


    # --------------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------------

    st.subheader(
        "💬 Ask About This Experiment"
    )

    communication_question = st.text_input(

        "Enter your question:",

        placeholder=(
            "Example: What is the procedure for "
            "Frequency Modulation?"
        ),

        key="communication_question"
    )


    if st.button(
        "Ask AI",
        key="communication_ask_ai"
    ):

        if not communication_question:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the laboratory manual..."
            ):

                answer = answer_from_lab_manual(
                    communication_question
                )


            if answer:

                st.subheader(
                    "🤖 AI Answer"
                )

                st.write(
                    answer
                )

            else:

                st.error(
                    "Gemini is currently unavailable. "
                    "Please try again later."
                )


    # --------------------------------------------------------
    # RAG DATABASE INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📚 RAG Knowledge Base"
    )

    st.write(
        f"Manual chunks loaded: "
        f"{len(rag_chunks)}"
    )

    st.write(
        f"FAISS vectors loaded: "
        f"{faiss_index.ntotal}"
    )

    st.write(
        "Procedure database: 7 experiments"
    )