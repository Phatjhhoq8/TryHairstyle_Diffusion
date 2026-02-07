import React, { useState, useEffect } from 'react';
import ImageUpload from './ImageUpload';
import ResultView from './ResultView';

const API_BASE_URL = 'http://localhost:8000';

const HairSwapper = () => {
    const [targetFile, setTargetFile] = useState(null);
    const [referenceFile, setReferenceFile] = useState(null);
    const [prompt, setPrompt] = useState("high quality, realistic hairstyle");
    const [useRefiner, setUseRefiner] = useState(false);
    const [resultUrl, setResultUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [taskId, setTaskId] = useState(null);

    // Polling logic
    useEffect(() => {
        let intervalId;

        if (taskId && isLoading) {
            intervalId = setInterval(async () => {
                try {
                    const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
                    const data = await response.json();

                    if (data.status === 'SUCCESS') {
                        setResultUrl(`http://localhost:8000${data.result_url}`);
                        setIsLoading(false);
                        setTaskId(null);
                    } else if (data.status === 'FAILURE') {
                        setError(data.error || 'Unknown error occurred');
                        setIsLoading(false);
                        setTaskId(null);
                    }
                } catch (err) {
                    console.error("Polling error", err);
                    setError("Failed to check status");
                    setIsLoading(false);
                    setTaskId(null);
                }
            }, 2000); // Poll every 2 seconds
        }

        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [taskId, isLoading]);

    // Helper to convert URL to File object
    const urlToFile = async (url, filename, mimeType) => {
        const res = await fetch(url);
        const buf = await res.arrayBuffer();
        return new File([buf], filename, { type: mimeType });
    };

    const handleRandomPair = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/random-pair`);
            const data = await res.json();

            if (data.target_url) {
                const targetUrl = `${API_BASE_URL}${data.target_url}`;
                const file = await urlToFile(targetUrl, "random_face.png", "image/png");
                setTargetFile(file);
            }

            if (data.hair_url) {
                const hairUrl = `${API_BASE_URL}${data.hair_url}`;
                const file = await urlToFile(hairUrl, "random_hair.png", "image/png");
                setReferenceFile(file);
            }
        } catch (err) {
            console.error(err);
            setError("Failed to load random images");
        } finally {
            setIsLoading(false);
        }
    };

    const handleGenerate = async () => {
        if (!targetFile || !referenceFile) {
            setError("Please upload both images.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setResultUrl(null);

        const formData = new FormData();
        formData.append('face_image', targetFile);
        formData.append('hair_image', referenceFile);
        formData.append('description', prompt);
        formData.append('use_refiner', useRefiner);

        try {
            const response = await fetch(`${API_BASE_URL}/generate`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to start generation');
            }

            const data = await response.json();
            setTaskId(data.task_id);
        } catch (err) {
            console.error("Generation error", err);
            setError(err.message);
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-[1280px] mx-auto px-4 py-6">

            {/* Header */}
            <div className="mb-6 border-b border-gray-700 pb-4">
                <h1 className="text-2xl font-bold text-gray-100">HairFusion</h1>
                <p className="text-gray-400 text-sm">Unofficial Implementation of HairFusion (AAAI2025)</p>
            </div>

            {/* Main Content using Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* Left Column: Inputs */}
                <div className="flex flex-col gap-4">

                    {/* Images Row */}
                    <div className="gradio-block p-4 bg-[#1f2937]">
                        <div className="flex justify-between items-center mb-3">
                            <span className="gradio-label text-base">Input Images</span>
                            <button
                                onClick={handleRandomPair}
                                disabled={isLoading}
                                className="text-xs bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded border border-gray-600 transition-colors"
                            >
                                🎲 Random Pair
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            <ImageUpload
                                id="target-upload"
                                label="Target Face"
                                onImageSelected={setTargetFile}
                                externalFile={targetFile}
                            />
                            <ImageUpload
                                id="ref-upload"
                                label="Reference Hair"
                                onImageSelected={setReferenceFile}
                                externalFile={referenceFile}
                            />
                        </div>
                    </div>

                    {/* Prompt Row */}
                    <div className="gradio-block p-4">
                        <label className="gradio-label">Editing Prompt</label>
                        <textarea
                            className="input-gradio h-24 resize-none font-mono text-sm"
                            placeholder="Describe the hairstyle change (e.g., 'blonde curly hair')"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                        />
                    </div>

                    {/* Settings Row */}
                    <div className="gradio-block p-4 bg-[#1f2937] flex items-center gap-3">
                        <input
                            type="checkbox"
                            id="useRefiner"
                            checked={useRefiner}
                            onChange={(e) => setUseRefiner(e.target.checked)}
                            className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500"
                        />
                        <div className="flex flex-col">
                            <label htmlFor="useRefiner" className="text-gray-200 font-medium cursor-pointer">
                                High Quality Mode (SDXL Refiner)
                            </label>
                            <span className="text-xs text-gray-400">
                                Slower (30-50%) but much better hair details.
                            </span>
                        </div>
                    </div>

                    {/* Button */}
                    <button
                        onClick={handleGenerate}
                        disabled={isLoading || !targetFile || !referenceFile}
                        className="btn-gradio-primary w-full py-3 text-lg"
                    >
                        {isLoading ? 'Running...' : 'Run Transfer 🚀'}
                    </button>

                </div>

                {/* Right Column: Output */}
                <div className="gradio-block p-4 min-h-[600px]">
                    <ResultView resultUrl={resultUrl} isLoading={isLoading} error={error} />
                </div>

            </div>

            {/* Footer / Citation */}
            <div className="mt-12 text-center text-xs text-gray-600 font-mono">
                Running on Localhost • Powered by FastAPI & React
            </div>
        </div>
    );
};

export default HairSwapper;
