import gradio as gr
from UI.infer_cae import infer_CAE_model
from UI.infer_lstm import infer_LSTM_model
from UI.infer_ml import infer_ML_model
from UI.infer_lstm_attention import infer_LSTM_attention_model
from Data.preprocess_for_input_infer import preprocess_text


model_infer_functions = {
    "LSTM Model": infer_LSTM_model,
    "Model manh vai ca loz": infer_CAE_model,
    "Machine Learning based Model":infer_ML_model,
    "LSTM with attention": infer_LSTM_attention_model
}

# Define the main inference function to call the appropriate model function
def infer_single_comment(comment, model_choice):
    # Call the appropriate inference function
    comment = preprocess_text(comment)

    return model_infer_functions[model_choice](comment)
# Set up Gradio Interface
demo = gr.Interface(
    fn=infer_single_comment,
    inputs=[
        "text",
        gr.Dropdown(choices=["LSTM Model", "Model manh vai ca loz","Machine Learning based Model","LSTM with attention"], label="Select Model")
    ],
    outputs="text",
    title="Vietnamese Aspect-Based Sentiment Analysis",
    description="Choose a model to predict sentiment for each aspect in a given Vietnamese review comment.",
    examples=[["Sản phẩm này có camera rất tốt nhưng giá hơi cao"]],
)

demo.launch(share=True)
