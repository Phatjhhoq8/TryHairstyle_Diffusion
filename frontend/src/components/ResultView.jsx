

import React from 'react';

const ResultView = ({ resultUrl, isLoading, error }) => {
    return (
        <div className="w-full h-full flex flex-col">
            <label className="gradio-label">Result</label>

            <div className="relative flex-1 w-full min-h-[500px] bg-[#374151] border border-gray-600 rounded-lg flex items-center justify-center overflow-hidden">

                {isLoading ? (
                    <div className="flex flex-col items-center gap-3">
                        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-orange-500"></div>
                        <p className="text-gray-400 text-sm animate-pulse">Running Inference...</p>
                        <p className="text-xs text-gray-500">This may take about 20s</p>
                    </div>
                ) : error ? (
                    <div className="text-center p-6 text-red-400">
                        <p className="font-bold">Error</p>
                        <p className="text-sm">{error}</p>
                    </div>
                ) : resultUrl ? (
                    <div className="relative w-full h-full">
                        <img src={resultUrl} alt="Result" className="w-full h-full object-contain bg-black/20" />
                        <a
                            href={resultUrl}
                            download="hairfusion_result.png"
                            className="absolute top-2 right-2 bg-gray-900/80 hover:bg-black text-white px-3 py-1 text-sm rounded transition-colors"
                        >
                            ⬇ Save
                        </a>
                    </div>
                ) : (
                    <div className="text-gray-500 font-mono text-sm">Output will appear here</div>
                )}
            </div>
        </div>
    );
};

export default ResultView;
