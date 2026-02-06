import React, { useRef, useState } from 'react';

const ImageUpload = ({ label, onImageSelected, id, externalFile }) => {
    const [preview, setPreview] = useState(null);
    const inputRef = useRef(null);

    // Handle external file changes (e.g. from Random button)
    React.useEffect(() => {
        if (externalFile) {
            handleFile(externalFile);
        }
    }, [externalFile]);

    const handleFile = (file) => {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
                onImageSelected(file);
            };
            reader.readAsDataURL(file);
        }
    };

    return (
        <div className="w-full">
            <label className="gradio-label">{label}</label>

            <div
                className="relative w-full aspect-[1/1] bg-[#374151] border-2 border-dashed border-gray-600 hover:border-gray-500 rounded-lg cursor-pointer transition-colors flex flex-col items-center justify-center overflow-hidden"
                onClick={() => inputRef.current?.click()}
                onDrop={(e) => {
                    e.preventDefault();
                    handleFile(e.dataTransfer.files[0]);
                }}
                onDragOver={(e) => e.preventDefault()}
            >
                {preview ? (
                    <div className="relative w-full h-full group">
                        <img src={preview} alt="Preview" className="w-full h-full object-contain bg-black/50" />
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                                className="bg-gray-800/80 text-white p-1 rounded hover:bg-gray-700"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setPreview(null);
                                    onImageSelected(null);
                                }}
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="text-center p-4">
                        <span className="text-4xl block mb-2">📷</span>
                        <span className="text-gray-400 text-sm">Drop Image Here<br />- or -<br />Click to Upload</span>
                    </div>
                )}
                <input
                    ref={inputRef}
                    id={id}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={(e) => handleFile(e.target.files[0])}
                />
            </div>
        </div>
    );
};

export default ImageUpload;
