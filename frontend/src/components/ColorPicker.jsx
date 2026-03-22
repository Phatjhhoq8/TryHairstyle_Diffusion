import { useState, useEffect } from 'react';
import { getColors } from '../api/hairApi';

export default function ColorPicker({ selectedColor, onColorChange, intensity, onIntensityChange }) {
  const [colors, setColors] = useState({});
  const [hexInput, setHexInput] = useState('');

  useEffect(() => {
    getColors()
      .then(setColors)
      .catch(() => {
        // Fallback — danh sách màu tĩnh nếu API chưa sẵn
        setColors({
          black: { hex: '#1a1a1a', label: 'Đen' },
          dark_brown: { hex: '#3b2213', label: 'Nâu đậm' },
          brown: { hex: '#6b3a2a', label: 'Nâu' },
          blonde: { hex: '#d4a54a', label: 'Vàng' },
          red: { hex: '#8b2500', label: 'Đỏ' },
          pink: { hex: '#d4608a', label: 'Hồng' },
          blue: { hex: '#2a52be', label: 'Xanh dương' },
          purple: { hex: '#6a0dad', label: 'Tím' },
          silver: { hex: '#c0c0c0', label: 'Bạc' },
        });
      });
  }, []);

  const handleSwatchClick = (name) => {
    if (selectedColor === name) {
      onColorChange('none'); // bỏ chọn
    } else {
      onColorChange(name);
      setHexInput('');
    }
  };

  const handleHexSubmit = () => {
    const hex = hexInput.trim();
    if (/^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$/.test(hex)) {
      onColorChange(hex);
    }
  };

  return (
    <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">Tuỳ chỉnh màu tóc</h3>
        {selectedColor && selectedColor !== 'none' && (
          <button
            onClick={() => onColorChange('none')}
            className="text-xs text-gray-400 hover:text-gray-600 transition"
          >
            Bỏ chọn màu
          </button>
        )}
      </div>

      {/* Color Swatches */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(colors).map(([name, info]) => (
          <button
            key={name}
            title={info.label}
            className={`color-swatch ${selectedColor === name ? 'active' : ''}`}
            style={{ backgroundColor: info.hex }}
            onClick={() => handleSwatchClick(name)}
          />
        ))}
      </div>

      {/* Hex Input */}
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={hexInput}
          onChange={(e) => setHexInput(e.target.value)}
          placeholder="#FF0000"
          className="flex-1 rounded-lg border border-gray-200 px-3 py-2 text-sm
                     focus:outline-none focus:ring-2 focus:ring-[#2d9b8e]/30 focus:border-[#2d9b8e]"
        />
        <button
          onClick={handleHexSubmit}
          className="px-3 py-2 bg-gray-200 text-gray-600 rounded-lg text-sm 
                     hover:bg-gray-300 transition"
        >
          Áp dụng
        </button>
      </div>

      {/* Intensity Slider */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <label className="text-xs text-gray-500">Cường độ màu</label>
          <span className="text-xs font-medium text-gray-600">{Math.round(intensity * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={intensity}
          onChange={(e) => onIntensityChange(parseFloat(e.target.value))}
          className="w-full accent-[#2d9b8e]"
        />
      </div>
    </div>
  );
}
