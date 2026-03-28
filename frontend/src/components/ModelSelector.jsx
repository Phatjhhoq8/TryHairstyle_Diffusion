import { useState } from 'react';

const MODELS = [
  {
    id: 'TryHairstyle',
    label: 'TryHairstyle',
    desc: '',
    icon: '',
  },
  {
    id: 'TryOnHairstyle',
    label: 'TryOnHairstyle',
    desc: '',
    icon: '',
  },
];

export default function ModelSelector({ value, onChange }) {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #f8f9fa, #e9ecef)',
      borderRadius: '12px',
      padding: '14px 16px',
      border: '1px solid #dee2e6',
    }}>
      <div style={{
        fontSize: '0.85rem',
        fontWeight: 600,
        color: '#495057',
        marginBottom: '10px',
      }}>
          Mô hình AI
      </div>
      <div style={{ display: 'flex', gap: '10px' }}>
        {MODELS.map((model) => (
          <button
            key={model.id}
            onClick={() => onChange(model.id)}
            style={{
              flex: 1,
              padding: '10px 12px',
              borderRadius: '10px',
              border: value === model.id
                ? '2px solid #2d9b8e'
                : '2px solid transparent',
              background: value === model.id
                ? 'linear-gradient(135deg, #e6f7f5, #d0f0ec)'
                : '#fff',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease',
              boxShadow: value === model.id
                ? '0 2px 8px rgba(45,155,142,0.2)'
                : '0 1px 3px rgba(0,0,0,0.08)',
            }}
          >
            <div style={{ fontSize: '1.2em', marginBottom: '4px' }}>
              {model.icon} {model.label}
            </div>
            <div style={{
              fontSize: '0.75rem',
              color: '#6c757d',
              lineHeight: 1.3,
            }}>
              {model.desc}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
