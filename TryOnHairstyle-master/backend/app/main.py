"""
HairFusion — Main Application Entry Point.
Provides both FastAPI REST endpoints and Gradio UI.
"""
import os
import sys
import shutil
import uuid

import cv2
import numpy as np
import torch

# Ensure backend package (sys.path) is initialized
import backend  # noqa: F401

from backend.app.config import DATA_DIR, DEVICE
from backend.app.services.preprocessing import PreprocessingService
from backend.app.services.diffusion import HairDiffusionService


class HairFusionApp:
    """Main application class orchestrating preprocessing and diffusion."""

    def __init__(self):
        print("=" * 50)
        print("Initializing HairFusion App...")
        print("=" * 50)

        print("[1/2] Loading Preprocessing Service...")
        self.preprocessor = PreprocessingService()

        print("[2/2] Loading Diffusion Service...")
        self.diffusion = HairDiffusionService()

        print("=" * 50)
        print("HairFusion App Ready!")
        print("=" * 50)

    def process(self, face_image, hair_image, steps=50, scale=5.0):
        """
        Full pipeline: preprocess + inpaint.

        Args:
            face_image: numpy array (RGB) - user's face
            hair_image: numpy array (RGB) - reference hairstyle
            steps: DDIM sampling steps
            scale: unconditional guidance scale

        Returns:
            tuple: (result_image, status_message)
        """
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        base_dir = os.path.join(DATA_DIR, session_id)

        try:
            # Cleanup previous run
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)

            # Save inputs
            face_path = os.path.join(base_dir, "images", "face.png")
            hair_path = os.path.join(base_dir, "images", "hair.png")

            cv2.imwrite(face_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(hair_path, cv2.cvtColor(hair_image, cv2.COLOR_RGB2BGR))

            # Write test pairs
            with open(os.path.join(base_dir, "test_pairs.txt"), "w") as f:
                f.write("face.png hair.png\n")

            # Preprocess
            print(f"[{session_id}] Preprocessing...")
            self.preprocessor.run_preprocess(face_path, session_id)
            self.preprocessor.run_preprocess(hair_path, session_id)
            self.preprocessor.run_make_agnostic("face.png", session_id)

            # Inference
            print(f"[{session_id}] Running inference...")
            results = self.diffusion.run_inference(
                data_root_dir=base_dir,
                steps=steps,
                scale=scale
            )

            if results:
                return results[0], "Success"
            else:
                return None, "No results generated"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"

        finally:
            # Cleanup temp data
            if os.path.exists(base_dir):
                try:
                    shutil.rmtree(base_dir)
                except:
                    pass


def run_gradio():
    """Launch Gradio web UI."""
    import gradio as gr

    app_engine = HairFusionApp()

    with gr.Blocks(title="HairFusion") as demo:
        gr.Markdown("## HairFusion: AI Hairstyle Transfer")

        with gr.Row():
            with gr.Column():
                input_face = gr.Image(label="Your Face", type="numpy")
                input_hair = gr.Image(label="Desired Hairstyle Reference", type="numpy")
                with gr.Row():
                    steps_slider = gr.Slider(10, 100, value=50, step=5, label="Steps")
                    scale_slider = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Guidance Scale")
                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Result")
                status_text = gr.Textbox(label="Status", interactive=False)

        def on_click(face, hair, steps, scale):
            if face is None or hair is None:
                return None, "Please upload both images."
            try:
                result, msg = app_engine.process(face, hair, int(steps), float(scale))
                return result, msg
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"Error: {str(e)}"

        run_btn.click(
            on_click,
            inputs=[input_face, input_hair, steps_slider, scale_slider],
            outputs=[output_image, status_text]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    run_gradio()
