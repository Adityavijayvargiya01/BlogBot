import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from ctransformers import AutoModelForCausalLM

# Function to get response from LLama 2 model using GPU acceleration
def getLLamaresponse(input_text, no_words, blog_style):
    # Speed-optimized configuration
    config = {
        'max_new_tokens': 128,        # Reduced for faster generation
        'temperature': 0.1,           # Slightly increased for faster sampling
        'top_p': 0.9,                # Added for faster sampling
        'top_k': 40,                 # Added for faster sampling
        'context_length': 512,        # Reduced for speed
        'threads': 6,                # Optimized for RTX 3050
        'gpu_layers': 24,            # Balanced for 4GB VRAM
        'batch_size': 8,             # Added for better GPU utilization
        'stream': True               # Enable streaming for faster apparent response
    }

    # Initialize the model with optimized configuration
    llm = CTransformers(
        model="models/llama-2-7b.ggmlv3.q8_0.bin",
        model_type="llama",
        config=config
    )

    # Optimized prompt template - more concise
    template = """Write a concise {no_words}-word blog for {blog_style} about {input_text}."""
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", 'no_words'],
        template=template
    )

    try:
        response = llm.invoke(prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            no_words=int(no_words)
        ))
        return response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Streamlit UI with performance monitoring
st.set_page_config(
    page_title="BlogBot",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("BlogBot 🤖")

# Performance info
st.info("⚡ Running with optimized GPU settings for faster generation")

# Input fields with default values for faster testing
input_text = st.text_input("Enter the Blog Topic", placeholder="e.g., Artificial Intelligence")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words', value='200')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Common People', 'Researchers', 'Data Scientist'),  # Reordered for common use
        index=0
    )

submit = st.button("Generate")

# Generate response with progress tracking
if submit:
    if not input_text:
        st.warning("Please enter a blog topic")
    else:
        with st.spinner('🚀 Generating your blog...'):
            response = getLLamaresponse(input_text, no_words, blog_style)
            if response:
                st.success("✅ Blog generated!")
                st.markdown(response, unsafe_allow_html=True)