import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import time
import psutil

def get_system_info():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    return cpu_percent, memory.percent

def optimize_model_config():
    """Optimize model configuration based on available hardware"""
    # Configuration optimized for your system (4GB VRAM, 16 threads CPU)
    config = {
        'max_new_tokens': 256,      # Increased for better quality
        'temperature': 0.7,         # Better creativity while maintaining coherence
        'top_p': 0.95,             # Slightly increased for better quality
        'top_k': 50,               # Balanced selection
        'context_length': 1024,     # Increased for better context understanding
        'threads': 16,             # Utilizing all CPU threads
        'gpu_layers': 20,          # Optimized for 4GB VRAM
        'batch_size': 1,           # Single batch for more stable generation
        'stream': True,            # Enable streaming
        'repetition_penalty': 1.1   # Prevent repetitive text
    }
    
    return config

def getLLamaresponse(input_text, no_words, blog_style):
    """Generate blog content with improved prompting and error handling"""
    config = optimize_model_config()
    
    try:
        # Initialize model with optimized settings
        llm = CTransformers(
            model="models/llama-2-7b.ggmlv3.q8_0.bin",
            model_type="llama",
            config=config
        )
        
        # Enhanced prompt template for better quality
        template = """
        Write a {no_words}-word blog post for {blog_style} about {input_text}.
        
        Guidelines:
        - Write in a clear, engaging style
        - Include relevant examples and explanations
        - Structure the content with clear paragraphs
        - Maintain appropriate technical depth for the audience
        - End with a meaningful conclusion
        
        Blog Content:
        """
        
        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "no_words"],
            template=template
        )
        
        # Generate response with timing
        start_time = time.time()
        response = llm.invoke(prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            no_words=no_words
        ))
        generation_time = time.time() - start_time
        
        return response, generation_time
    
    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        return None, 0

def main():
    st.set_page_config(
        page_title="Enhanced BlogBot",
        page_icon='🤖',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Improved UI with sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        blog_style = st.selectbox(
            'Target Audience',
            ('Common People', 'Researchers', 'Data Scientists', 'Business Professionals'),
            index=0,
            help="Select your target audience for appropriate content style"
        )
        
        no_words = st.slider(
            'Word Count',
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="Select the desired length of your blog post"
        )
        
        # System monitoring
        st.header("📊 System Monitor")
        cpu_percent, memory_percent = get_system_info()
        st.progress(cpu_percent/100, text=f"CPU Usage: {cpu_percent}%")
        st.progress(memory_percent/100, text=f"Memory Usage: {memory_percent}%")

    # Main content area
    st.header("Enhanced BlogBot 🤖")
    st.markdown("""
    Generate high-quality blog posts optimized for your target audience.
    Enter your topic below and click 'Generate' to create your blog post.
    """)
    
    input_text = st.text_area(
        "Blog Topic",
        height=100,
        placeholder="Enter your blog topic here. Be specific for better results.",
        help="Provide a clear topic or question for your blog post"
    )
    
    if st.button("Generate Blog", type="primary"):
        if not input_text:
            st.warning("Please enter a blog topic to continue.")
            return
            
        with st.spinner('🚀 Generating your blog post...'):
            response, generation_time = getLLamaresponse(input_text, no_words, blog_style)
            
            if response:
                st.success(f"✅ Blog generated in {generation_time:.2f} seconds!")
                
                # Display the blog post in a clean format
                st.markdown("---")
                st.markdown("### Generated Blog Post")
                st.markdown(response)
                st.markdown("---")
                
                # Provide export options
                st.download_button(
                    label="Download Blog Post",
                    data=response,
                    file_name="generated_blog.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()