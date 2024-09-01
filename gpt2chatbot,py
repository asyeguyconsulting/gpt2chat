import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Streamlit app
st.title("GPT-2 Text Generator")
st.write("Enter a prompt to generate text using the GPT-2 model from Hugging Face.")

# User input
user_prompt = st.text_input("Enter your prompt:")

# Button to generate text
if st.button("Generate Response"):
    if user_prompt:
        try:
            # Encode the input prompt and generate a response
            input_ids = tokenizer.encode(user_prompt, return_tensors='pt')
            with torch.no_grad():
                output = model.generate(input_ids, max_length=150, num_return_sequences=1)

            # Decode the output to get the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Display the generated response
            st.write("**Response from GPT-2:**")
            st.write(generated_text)
        except Exception as e:
            st.error("An error occurred while generating the response.")
            st.write(f"Error details: {e}")
    else:
        st.warning("Please enter a prompt.")
