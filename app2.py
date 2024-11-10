import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Ensure you have the correct model path if downloading from Hugging Face
MODEL_NAME = "models/llama-2-7b.ggmlv3.q8_0.bin"

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

model, tokenizer = load_model()

# Function to get response from LLaMA 2 model using GPU
def get_llama_response(input_text, no_words, blog_style):
    prompt = f"Write a blog for {blog_style} job profile on the topic '{input_text}' within {no_words} words."

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=int(no_words),
        temperature=0.01,
        top_p=0.9,
        repetition_penalty=1.2
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.set_page_config(page_title="BlogBot", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.header("BlogBot 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    if input_text and no_words.isdigit():
        response = get_llama_response(input_text, no_words, blog_style)
        st.markdown(response, unsafe_allow_html=True)
    else:
        st.warning("Please enter valid inputs.")
