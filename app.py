import torch
import gradio as gr

import models


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


model, tokenizer = models.load_model()
state_dict = torch.load("output/bert_spam.pth", weights_only=True)
model.load_state_dict(state_dict)
model.to(device)


def inference(subject, message):
    texts = f'SUBJECT: "{subject}"\nMESSAGE: "{message}"'
    encoded_inputs = tokenizer(
        [texts],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(encoded_inputs)
        logits = torch.softmax(outputs, dim=1).cpu().numpy()
    return logits[0]


with gr.Blocks() as demo:
    gr.Markdown("# BERT 模型推論 UI")
    gr.Markdown("請輸入主旨 (subject) 和內文 (message)，以獲得模型輸出的機率")

    with gr.Row():
        with gr.Column():
            subject_input = gr.Textbox(
                label="Subject", placeholder="Enter the subject here..."
            )
            message_input = gr.Textbox(
                label="Message",
                placeholder="Enter the message here...",
                max_lines=10,
            )
            submit_button = gr.Button("Submit")

        with gr.Row():
            bar_chart = gr.Plot(label="Probability Visualization")

    def visualize_prob(subject, message):
        logits = inference(subject, message)
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Probability",
                    y=["Ham", "Spam"],
                    x=logits,
                    orientation="h",
                    marker_color=["blue", "orange"],
                )
            ]
        )
        fig.update_layout(
            title="Probability Visualization",
            xaxis_title="Probability",
            yaxis_title="Class",
            xaxis=dict(range=[0, 1]),
        )

        return fig

    submit_button.click(
        fn=visualize_prob, inputs=[subject_input, message_input], outputs=bar_chart
    )

# 啟動介面
demo.launch(server_name="0.0.0.0")
