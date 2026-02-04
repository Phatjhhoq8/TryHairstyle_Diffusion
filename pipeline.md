graph TD
    %% Inputs
    Target["Ảnh Gốc (Target)"] --> InsightFace["InsightFace<br>(Face Analysis)"]
    Target --> Seg["Segmentation<br>(BiSeNet/FaceParsing)"]
    Target --> Depth["ControlNet Preprocessor<br>(Depth Estimation)"]
    Ref["Ảnh Mẫu (Reference)"] --> CLIP["CLIP Image Encoder"]

    %% Preprocessing Data
    InsightFace -->|ID Embedding<br>Keypoints| InstantID["Adapter: InstantID<br>(Giữ Identity)"]
    Seg -->|Hair Mask + Dilate| InpaintMask["Inpainting Mask"]
    Depth -->|Depth Map| ControlNet["Adapter: ControlNet<br>(Giữ Shape/Pose)"]
    CLIP -->|Style Embedding| IPAdapter["Adapter: IP-Adapter Plus<br>(Lấy Style Tóc)"]

    %% Inference Core
    InstantID --> SDXL["SDXL Inpainting Model"]
    InpaintMask --> SDXL
    ControlNet --> SDXL
    IPAdapter --> SDXL

    %% Models Logic
    SDXL -->|"Denoising Loop<br>(Guided by Adapters)"| Latent["Latent Output"]
    Latent --> Decode["VAE Decode"]

    %% Post-processing
    Decode --> RawImg["Ảnh thô"]
    RawImg --> Restore["Face Restore<br>(CodeFormer)"]
    Restore --> Final["Ảnh Kết Quả"]

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef model fill:#dfd,stroke:#333,stroke-width:2px;
    classDef output fill:#ff9,stroke:#333,stroke-width:4px;

    class Target,Ref input;
    class InsightFace,Seg,Depth,CLIP,Restore,Decode process;
    class InstantID,ControlNet,IPAdapter,SDXL model;
    class Final output;