import gradio as gr
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from aptatrans_pipeline import AptaTransPipeline
import torch

def predict_api_score(aptamer, target):
    try:
        # Initialize pipeline with correct data path
        pipeline = AptaTransPipeline(
            dim=128,
            mult_ff=2,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            load_best_pt=False,
            load_best_model=True,
            save_name='default',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            seed=1004,
            data_dir=os.path.join(project_root, 'data')
        )
        
        # Make prediction
        score = pipeline.inference(aptamer, target)
        return f"Predicted API Score: {score[0][0]:.4f}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def recommend_aptamers(target):
    try:
        # Initialize pipeline with correct data path
        pipeline = AptaTransPipeline(
            dim=128,
            mult_ff=2,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            load_best_pt=False,
            load_best_model=True,
            save_name='default',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            seed=1004,
            data_dir=os.path.join(project_root, 'data')
        )
        
        # Get recommendations
        results = pipeline.recommend(target, n_aptamers=5, depth=40, iteration=1000)
        
        # Format output
        output = "Recommended Aptamers:\n\n"
        for idx, result in results.items():
            output += f"Candidate {idx + 1}:\n"
            output += f"Sequence: {result['candidate']}\n"
            output += f"Score: {result['score'].item():.4f}\n\n"
        
        return output
    except Exception as e:
        return f"Error during recommendation: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="AptaTrans") as demo:
    gr.Markdown("# AptaTrans - Aptamer-Protein Interaction Prediction")
    
    with gr.Tab("Predict API Score"):
        gr.Markdown("## Predict Aptamer-Protein Interaction Score")
        with gr.Row():
            with gr.Column():
                aptamer_input = gr.Textbox(label="Aptamer Sequence", placeholder="Enter aptamer sequence...")
                target_input = gr.Textbox(label="Target Protein Sequence", placeholder="Enter target protein sequence...")
                predict_btn = gr.Button("Predict Score")
            with gr.Column():
                output_score = gr.Textbox(label="Prediction Result")
        
        predict_btn.click(
            fn=predict_api_score,
            inputs=[aptamer_input, target_input],
            outputs=output_score
        )
    
    with gr.Tab("Recommend Aptamers"):
        gr.Markdown("## Recommend Candidate Aptamers")
        with gr.Row():
            with gr.Column():
                target_input_recommend = gr.Textbox(label="Target Protein Sequence", placeholder="Enter target protein sequence...")
                recommend_btn = gr.Button("Recommend Aptamers")
            with gr.Column():
                output_recommend = gr.TextArea(
                    label="Recommendation Results",
                    lines=15,  # Set number of visible lines
                    max_lines=50  # Set maximum number of lines
                )
        
        recommend_btn.click(
            fn=recommend_aptamers,
            inputs=target_input_recommend,
            outputs=output_recommend
        )

if __name__ == "__main__":
    demo.launch(share=True)
