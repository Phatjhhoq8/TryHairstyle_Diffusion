export default function DrawButton({ onClick, loading, disabled, onQuickColor, colorLoading, hasResult }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3">
      <button
        className="draw-btn"
        onClick={onClick}
        disabled={loading || disabled}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Đang xử lý...
          </>
        ) : (
          'VẼ TÓC'
        )}
      </button>
      <p className="text-xs text-gray-400 text-center">Nhấn để tạo kết quả</p>

      {/* Nút đổi màu nhanh — chỉ hiện khi đã có ảnh kết quả */}
      {hasResult && (
        <button
          className="quick-color-btn"
          onClick={onQuickColor}
          disabled={colorLoading || loading}
        >
          {colorLoading ? (
            <>
              <span className="spinner spinner-sm" />
              Đang đổi màu...
            </>
          ) : (
            'ĐỔI MÀU NHANH'
          )}
        </button>
      )}
    </div>
  );
}
