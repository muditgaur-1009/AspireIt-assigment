# Sentiment Analysis with GPT-2

This Streamlit application performs sentiment analysis on user conversations using the GPT-2 language model. The app generates AI responses to user inputs, analyzes the sentiment of these responses, and provides visual and PDF reports of the analysis.

## Features

- **GPT-2 Response Generation:** Generates AI responses to user inputs using the GPT-2 model.
- **Sentiment Analysis:** Analyzes the sentiment of the AI responses (positive, negative, neutral).
- **Data Visualization:** Creates charts showing the distribution of sentiment polarity and subjectivity.
- **PDF Report Generation:** Generates a PDF report of the conversation and sentiment analysis results.

## Installation

To run this application, you need to have Python installed. Follow the steps below to set up the application:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/muditgaur-1009/AspireIt-assigment.git
    cd Aspire-Assignment
    ```

2. **Install the required packages:**

    ```sh
    pip install streamlit torch transformers textblob matplotlib seaborn pandas reportlab
    ```

## Usage

1. **Run the Streamlit application:**

    ```sh
    streamlit run main.py
    ```

2. **Interact with the application:**

    - Type your message in the input box and press Enter.
    - The AI will generate a response, and its sentiment will be analyzed.
    - View the conversation history and sentiment analysis results.
    - Type 'exit' to finish the conversation and generate the report.
    - Click the "Generate Report" button to download a PDF report of the conversation and sentiment analysis.

## Code Structure

- **main.py:** The main script that runs the Streamlit application.
- **requirements.txt:** Lists the required Python packages.

## Functions

### load_model_and_tokenizer()

Loads the GPT-2 model and tokenizer from the Hugging Face library.

### generate_response(prompt)

Generates a response from the GPT-2 model based on the user input prompt.

### analyze_sentiment(response_text)

Analyzes the sentiment of the response text using TextBlob.

### create_charts(df)

Creates charts for sentiment polarity and subjectivity distribution, and a pie chart for sentiment distribution.

### generate_pdf_report(df, charts)

Generates a PDF report containing the conversation data and charts.

## Example

![Screenshot of the app](Aspire-Assignment\Screenshot 2024-09-26 194528.png)



## Contact

For any inquiries or issues, please contact [Mudit Gaur](mailto:muditgaur1009@gmail.com).
