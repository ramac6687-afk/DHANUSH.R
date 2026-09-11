import streamlit as st
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -----------------------------------
# Load API key
# -----------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# -----------------------------------
# Gemini models
# -----------------------------------

models_to_try = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]

# -----------------------------------
# Page
# -----------------------------------

st.title("AI Lab Assistant")

st.write(
    "Intelligent Electronics Experiment Diagnostic Assistant"
)

# -----------------------------------
# Load knowledge base
# -----------------------------------

with open("knowledge_base.json", "r") as file:
    knowledge_base = json.load(file)

# -----------------------------------
# Load fault dataset
# -----------------------------------

with open("fault_dataset.json", "r") as file:
    fault_dataset = json.load(file)

# -----------------------------------
# Select experiment
# -----------------------------------

experiment = st.selectbox(
    "Select your experiment:",
    [
        "Band Pass Filter",
        "Voltage Divider",
        "LED Circuit"
    ]
)

experiment_key = experiment.lower().replace(" ", "_")

data = knowledge_base[experiment_key]

# -----------------------------------
# Experiment information
# -----------------------------------

st.subheader("Experiment Information")

st.write("**Formula:**", data["formula"])

st.write("**Expected Result:**", data["expected_result"])

st.subheader("Common Faults")

for fault in data["common_faults"]:
    st.write("•", fault)

# -----------------------------------
# Circuit photo
# -----------------------------------

st.subheader("Circuit Photo")

uploaded_image = st.file_uploader(
    "Upload a photo of your circuit:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    st.image(
        uploaded_image,
        caption="Uploaded Circuit",
        width="stretch"
    )

# -----------------------------------
# Measurements
# -----------------------------------

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

# -----------------------------------
# Observed result
# -----------------------------------

st.subheader("Observed Result")

observed_result = st.text_input(
    "Enter your measured result:",
    placeholder="Example: 15 kHz"
)

# -----------------------------------
# Circuit Diagnosis
# -----------------------------------

st.divider()

st.header("🔍 Circuit Diagnosis")

if st.button("Diagnose Circuit"):

    if not uploaded_image:

        st.warning("Please upload a circuit photo.")

    elif not observed_result:

        st.warning("Please enter your observed result.")

    else:

        with st.spinner("AI is analyzing your circuit..."):

            # -----------------------------------
            # Prepare fault examples
            # -----------------------------------

            training_cases = fault_dataset.get(
                experiment_key,
                []
            )

            training_information = json.dumps(
                training_cases,
                indent=2
            )

            # -----------------------------------
            # Prepare circuit image
            # -----------------------------------

            image_part = types.Part.from_bytes(
                data=uploaded_image.getvalue(),
                mime_type=uploaded_image.type
            )

            # -----------------------------------
            # AI diagnosis prompt
            # -----------------------------------

            prompt = f"""
You are an electronics laboratory diagnostic assistant.

Your task is to diagnose faults in a student's electronics experiment.

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
The following cases are examples from the experiment's fault
knowledge dataset.

{training_information}

IMPORTANT INSTRUCTIONS:

1. Analyze the student's actual circuit image.
2. Analyze the student's measurements.
3. Use the reference fault cases as diagnostic knowledge.
4. Do not automatically select a reference fault just because it exists.
5. Compare the student's symptoms with the possible faults.
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

            # -----------------------------------
            # Ask Gemini
            # -----------------------------------

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

            # -----------------------------------
            # Display diagnosis
            # -----------------------------------

            if response:

                st.subheader("🤖 AI Diagnosis")

                st.write(response.text)

            else:

                st.error(
                    "Gemini is currently unavailable. "
                    "Please try again later."
                )

# -----------------------------------
# Ask AI Chat
# -----------------------------------

st.divider()

st.header("💬 Ask AI About This Experiment")

st.write(
    "Ask a question related to the selected experiment."
)

question = st.text_input(
    "Your question:",
    placeholder="Example: Why is my output voltage different?"
)

if st.button("Ask AI"):

    if not question:

        st.warning("Please enter a question.")

    else:

        with st.spinner("AI is thinking..."):

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

                st.subheader("🤖 AI Answer")

                st.write(response.text)

            else:

                st.error(
                    "Gemini is currently unavailable. "
                    "Please try again later."
                )