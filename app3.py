import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLama 2 model using GPU acceleration
def getLLamaresponse(input_text, no_words, blog_style):
    # Configure model to use CUDA (GPU)
    config = {
        'max_new_tokens': 256,
        'temperature': 0.01,
        'gpu_layers': 50,  # Number of layers to offload to GPU
        'context_length': 2048,  # Adjust based on your needs
        'threads': 8,  # Adjust based on your CPU
        'batch_size': 512,  # Adjust based on your GPU memory
        'use_cuda': True,  # Enable CUDA
        'cuda_device': 0  # Use first CUDA device
    }

    # Initialize the model with GPU configuration
    llm = CTransformers(
        model="models/llama-2-7b.ggmlv3.q8_0.bin",
        model_type='llama',
        config=config
    )

    # Prompt Template
    template = """
    Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", 'no_words'],
        template=template
    )

    # Generate the response from the LLama 2 model
    response = llm.invoke(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=int(no_words)
    ))
    
    return response

# Streamlit UI
st.set_page_config(
    page_title="BlogBot",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("BlogBot 🤖")

# Input fields
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

submit = st.button("Generate")

# Generate response
if submit:
    with st.spinner('Generating blog...'):
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.markdown(response, unsafe_allow_html=True)