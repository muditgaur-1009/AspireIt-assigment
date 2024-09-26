import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Load the GPT-2 model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Function to generate response using GPT-2
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function for sentiment analysis
def analyze_sentiment(response_text):
    analysis = TextBlob(response_text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return {"response": response_text, "polarity": polarity, "subjectivity": subjectivity, "sentiment": sentiment}

# Function to create charts
def create_charts(df):
    charts = []
    
    # Plotting polarity
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['polarity'], bins=10, kde=True, ax=ax)
    ax.set_title('Distribution of Sentiment Polarity')
    ax.set_xlabel('Polarity')
    ax.set_ylabel('Frequency')
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    charts.append(('polarity.png', img_buf))
    plt.close(fig)
    
    # Plotting subjectivity
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['subjectivity'], bins=10, kde=True, color='orange', ax=ax)
    ax.set_title('Distribution of Subjectivity')
    ax.set_xlabel('Subjectivity')
    ax.set_ylabel('Frequency')
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    charts.append(('subjectivity.png', img_buf))
    plt.close(fig)
    
    # Sentiment pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    sentiment_counts = df['sentiment'].value_counts()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightblue'])
    ax.set_title('Sentiment Distribution')
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    charts.append(('sentiment_pie.png', img_buf))
    plt.close(fig)
    
    return charts

# Function to generate PDF report
def generate_pdf_report(df, charts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Sentiment Analysis Report")

    # Add charts
    y_position = height - 100
    for chart_name, chart_data in charts:
        img = ImageReader(chart_data)
        c.drawImage(img, 50, y_position - 300, width=500, height=300)
        y_position -= 350
        if y_position < 50:
            c.showPage()
            y_position = height - 100

    # Add conversation data
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Conversation Data:")
    y_position -= 20
    c.setFont("Helvetica", 10)
    for index, row in df.iterrows():
        c.drawString(50, y_position, f"Response: {row['response'][:100]}...")
        y_position -= 15
        c.drawString(50, y_position, f"Sentiment: {row['sentiment']}, Polarity: {row['polarity']:.2f}, Subjectivity: {row['subjectivity']:.2f}")
        y_position -= 20
        if y_position < 50:
            c.showPage()
            y_position = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# Streamlit app
def main():
    st.title("Sentiment Analysis with GPT-2")

    conversations = []

    st.write("Type your message below and press Enter. Type 'exit' to finish the conversation and generate the report.")

    # Initialize session state for conversations if it doesn't exist
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []

    # Text input for user message
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        if user_input.lower() == "exit":
            st.write("Generating report...")
        else:
            ai_response = generate_response(user_input)
            st.write(f"AI: {ai_response}")
            sentiment = analyze_sentiment(ai_response)
            st.session_state.conversations.append(sentiment)

    # Display conversation history
    if st.session_state.conversations:
        st.subheader("Conversation History:")
        for idx, conv in enumerate(st.session_state.conversations, 1):
            st.write(f"{idx}. User: {conv['response']}")
            st.write(f"   Sentiment: {conv['sentiment']}, Polarity: {conv['polarity']:.2f}, Subjectivity: {conv['subjectivity']:.2f}")

    # Generate report button
    if st.button("Generate Report"):
        if st.session_state.conversations:
            df = pd.DataFrame(st.session_state.conversations)
            charts = create_charts(df)
            pdf_buffer = generate_pdf_report(df, charts)
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No conversations to analyze. Please have a conversation first.")

if __name__ == "__main__":
    main()
