import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Global model configuration
MODEL_PATH = "models/llama-2-7b.ggmlv3.q8_0.bin"
MODEL_CONFIG = {
    'max_new_tokens': 256,
    'temperature': 0.01,
    'gpu_layers': 20,     # Optimized for 4GB VRAM
    'threads': 16,        # Utilizing your 16 CPU threads
    'batch_size': 1
}

# Initialize model once at startup
@st.cache_resource
def initialize_model():
    return CTransformers(
        model=MODEL_PATH,
        model_type='llama',
        config=MODEL_CONFIG
    )

def getLLamaresponse(llm, input_text, no_words, blog_style):
    template = """
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", 'no_words'],
        template=template
    )
    
    response = llm.invoke(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    ))
    return response

# Streamlit UI
st.set_page_config(
    page_title="BlogBot",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# Initialize model
llm = initialize_model()

st.header("BlogBot 🤖")
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words', value='300')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

submit = st.button("Generate")

if submit:
    if not input_text:
        st.warning("Please enter a blog topic")
    else:
        with st.spinner('Generating...'):
            response = getLLamaresponse(llm, input_text, no_words, blog_style)
            st.markdown(response, unsafe_allow_html=True)