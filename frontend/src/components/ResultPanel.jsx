export default function ResultPanel({ resultUrl, status, error, onClear }) {
  return (
    <div className="result-panel flex flex-col gap-3 h-full">
      <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Ảnh kết quả</h3>

      {status === 'PROCESSING' && (
        <div className="flex flex-col items-center gap-3 py-8">
          <div className="progress-bar w-full max-w-[200px]">
            <div className="progress-bar-fill" />
          </div>
          <p className="text-sm text-gray-500">Đang xử lý, vui lòng chờ...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {resultUrl && status !== 'PROCESSING' && (
        <div className="flex flex-col gap-3">
          <div className="relative group">
            <img src={resultUrl} alt="Kết quả" className="rounded-lg w-full object-cover" />
            <button
              onClick={onClear}
              className="absolute top-2 right-2 bg-gray-800/50 hover:bg-gray-800/80 text-white rounded-full p-1.5 opacity-0 group-hover:opacity-100 transition shadow-sm backdrop-blur-sm"
              title="Xóa ảnh"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
          </div>
          <a
            href={resultUrl}
            download="hair_result.png"
            className="inline-flex items-center justify-center gap-2 bg-[#2d9b8e] text-white 
                       rounded-lg px-4 py-2.5 text-sm font-medium hover:bg-[#1a7a6d] transition w-full"
          >
            Tải xuống kết quả
          </a>
        </div>
      )}

      {!resultUrl && status !== 'PROCESSING' && !error && (
        <div className="flex items-center justify-center py-16 text-gray-300 text-sm">
          Kết quả sẽ hiển thị ở đây
        </div>
      )}
    </div>
  );
}
