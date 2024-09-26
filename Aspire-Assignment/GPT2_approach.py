import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import io

# Load the GPT-2 model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Function to generate response using Hugging Face GPT-2 model
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, do_sample=True, top_p=0.95, top_k=50, pad_token_id=tokenizer.eos_token_id)
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

# Streamlit app
def main():
    st.title("Sentiment Analysis with GPT-2")

    if 'conversations' not in st.session_state:
        st.session_state.conversations = []

    user_input = st.text_input("You:", key="user_input")

    if user_input:
        if user_input.lower() == "exit":
            st.write("Generating report...")
        else:
            gpt_response = generate_response(user_input)
            st.write(f"GPT-2: {gpt_response}")
            sentiment = analyze_sentiment(gpt_response)
            st.session_state.conversations.append(sentiment)

    # Display conversation history in a scrollable box
    st.subheader("Conversation History:")
    conversation_container = st.container()
    with conversation_container:
        if st.session_state.conversations:
            for idx, conv in enumerate(st.session_state.conversations, 1):
                st.write(f"{idx}. User: {conv['response']}")
                st.write(f"   Sentiment: {conv['sentiment']}, Polarity: {conv['polarity']:.2f}, Subjectivity: {conv['subjectivity']:.2f}")

    # Generate report button
    if st.button("Generate Report"):
        if st.session_state.conversations:
            df = pd.DataFrame(st.session_state.conversations)
            charts = create_charts(df)

            # Display charts
            for chart_name, chart_data in charts:
                st.image(chart_data, use_column_width=True)
        else:
            st.warning("No conversations to analyze. Please have a conversation first.")

if __name__ == "__main__":
    main()
