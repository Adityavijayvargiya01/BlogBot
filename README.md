# BlogBot

BlogBot is a Streamlit application that leverages the power of the LLaMA 2 language model to generate high-quality, customized blog posts on demand. With a user-friendly interface, you can input a blog topic, specify the desired word count, and select the target audience (Researchers, Data Scientists, or Common People). The application then utilizes the LLaMA 2 model to generate a tailored blog post based on your preferences.

## Features

- **Intuitive User Interface**: The Streamlit framework provides a sleek and intuitive layout for interacting with the application.
- **Customizable Blog Posts**: Generate blog posts tailored to specific topics, word counts, and target audiences.
- **LLaMA 2 Language Model**: Powered by the highly capable LLaMA 2 language model, trained on vast amounts of text data.
- **Easy Integration**: The modular design and the use of popular libraries like Streamlit and LangChain facilitate future enhancements and integrations.

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/BlogBot.git
```

2. Navigate to the project directory:

```
cd BlogBot
```

3. Create a virtual environment (optional but recommended):

```
conda create -n venv python==3.12 -y && conda activate venv
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Download the LLaMA 2 model from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-GGML/tree/main) or any other model of your choice  and place it in the `models` directory and MODEL_PATH in `app.py` file.

2. Run the Streamlit application:

```
streamlit run app.py
```

3. In your web browser, navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Enter the blog topic, desired word count, and select the target audience from the dropdown menu.

5. Click the "Generate" button to generate the customized blog post.

6. The generated blog post will be displayed in the application window.

## Note
- The LLaMA 2 model is quite large, so it may take some time to generate the blog post depending on your hardware specifications.

- The quality of the generated blog post may vary based on the input topic, word count, and target audience. Experiment with different settings to find the best results.

- For Better Results , Choose model with more parameters. 

## Requirements

The required dependencies are listed in the `requirements.txt` file. Here are the main dependencies:

- sentence-transformers
- uvicorn
- ctransformers
- langchain
- python-box
- streamlit
- langchain_community

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


## Acknowledgments

- The [Streamlit](https://streamlit.io/) team for creating the amazing Streamlit framework.
- The [LangChain](https://github.com/hwchase17/langchain) and [LangChain Community](https://github.com/hwchase17/langchain-community) projects for providing language model integration tools.
- The developers of the LLaMA 2 language model for their contributions to the field of natural language processing.