import { useRef } from 'react';

export default function ImageUpload({ label, image, onImageSelect, accept = 'image/*' }) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect(file);
      e.target.value = ''; // cho phép chọn lại cùng file
    }
  };

  const handleClear = (e) => {
    e.stopPropagation();
    onImageSelect(null);
  };

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">{label}</h3>
      <div
        className={`upload-zone ${image ? 'has-image' : ''}`}
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {image ? (
          <>
            <img src={typeof image === 'string' ? image : URL.createObjectURL(image)} alt="Preview" />
            <button
              onClick={handleClear}
              className="absolute top-2 right-2 bg-black/50 text-white rounded-full w-7 h-7 
                         flex items-center justify-center text-sm hover:bg-black/70 transition"
            >
              ✕
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-400 select-none p-6">
            <div className="w-12 h-12 rounded-full bg-[#2d9b8e]/10 flex items-center justify-center">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2d9b8e" strokeWidth="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17,8 12,3 7,8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <span className="text-sm text-center leading-relaxed">
              Kéo thả ảnh vào đây<br />
              <span className="text-gray-300">hoặc bấm để chọn file (PNG, JPG)</span>
            </span>
          </div>
        )}
      </div>
      <input ref={inputRef} type="file" accept={accept} onChange={handleChange} className="hidden" />
    </div>
  );
}
