import { useState } from 'react';

export default function FaceSelector({ faces, title, onConfirm, onCancel }) {
  const [selectedIndex, setSelectedIndex] = useState(null);

  if (!faces || faces.length === 0) return null;

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-semibold text-gray-800 mb-1">{title}</h3>
        <p className="text-sm text-gray-500 mb-4">
          Phát hiện {faces.length} khuôn mặt. Chọn 1 để tiếp tục xử lý.
        </p>

        <div className="face-grid">
          {faces.map((face, i) => (
            <div
              key={face.face_id ?? i}
              className={`face-card ${selectedIndex === i ? 'selected' : ''}`}
              onClick={() => setSelectedIndex(i)}
            >
              <img
                src={face.cropped_image_url}
                alt={`Khuôn mặt ${i + 1}`}
              />
              <div className="px-2 py-1.5 text-center">
                <span className="text-xs text-gray-500">
                  {face.confidence ? `${Math.round(face.confidence * 100)}%` : `#${i + 1}`}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-3 mt-5">
          <button
            onClick={() => selectedIndex !== null && onConfirm(faces[selectedIndex])}
            disabled={selectedIndex === null}
            className="flex-1 bg-[#2d9b8e] text-white rounded-lg px-4 py-2.5 text-sm font-medium
                       hover:bg-[#1a7a6d] transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Xác nhận chọn
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2.5 bg-gray-100 text-gray-600 rounded-lg text-sm 
                       hover:bg-gray-200 transition"
          >
            Huỷ
          </button>
        </div>
      </div>
    </div>
  );
}
